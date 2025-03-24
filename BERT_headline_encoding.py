# BERT-only model, which functions as a sentiment analysis classifier, predicting price movement based solely on headlines
# code source: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb#scrollTo=797b2WHJqUgZ

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
from datasets import Dataset
import glob


def preprocess_data(dataset):
    """Tokenize headlines and prepare labels for single-label classification"""
    text = dataset["headline"]
    encoding = tokenizer(
        text, 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )
    encoding["labels"] = dataset["label"]
    return encoding

def compute_metrics(p: EvalPrediction):
    """Metrics for single-label classification"""
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    probs = torch.nn.functional.softmax(torch.Tensor(preds), dim=-1).numpy()
    y_pred = np.argmax(probs, axis=1)
    y_true = p.label_ids
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, probs, multi_class="ovr")
    }


if __name__ == "__main__":
    # construct huggingface dataset from dataframe
    dataframe = pd.read_csv("all_batches.csv", encoding="latin1")

    # integrate "up", "stay" and "down" into one "label"
    dataframe["label"] = dataframe[["up", "stay", "down"]].idxmax(axis=1) # ["up", "stay", "down"] in dataframe["label"]
    label2id = {"up": 0, "stay": 1, "down": 2}
    id2label = {v: k for k, v in label2id.items()}
    dataframe["label"] = dataframe["label"].map(label2id) # [0, 1, 2] in dataframe["label"]

    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # tokenize headline in each record
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=["headline", "up", "stay", "down"])

    # pretrained BERT model for classification
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                               problem_type="single_label_classification", 
                                                               num_labels=3,
                                                               id2label=id2label,
                                                               label2id=label2id)
    
    # training configuration
    batch_size = 32
    metric_name = "f1_macro"
    args = TrainingArguments(
        f"bert-finetuned-sem_eval-english-v2", # checkpoint file is stored into this folder every training epoch
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

    # training and testing
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # examine check checkpoint files
    checkpoint_dirs = glob.glob("bert-finetuned-sem_eval-english-v2/checkpoint-*")
    if checkpoint_dirs:
        latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))[-1]
        train_output = trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        train_output = trainer.train()

    # record training and testing results
    test_output = trainer.evaluate()
    with open("results-v2.txt", "w", encoding="utf-8") as f:
        f.write("training results:\n")
        f.write(str(train_output) + "\n\n")
        f.write("testing results:\n")
        f.write(str(test_output) + "\n")