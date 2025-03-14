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


def preprocess_data(dataset):
    """
    Tokenize the "headline" feature for each record in dataset.
    """
    text = dataset["headline"]
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    labels_batch = {k: dataset[k] for k in dataset.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding

def multi_label_metrics(predictions, labels, threshold=0.5):
    """
    Auxiliary function of compute_metrics.
    """
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    """
    Compute f1, roc_auc, accuracy metrics of model predictions.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


if __name__ == "__main__":
    # construct huggingface dataset from dataframe
    dataframe = pd.read_csv("headlines_labels.csv", encoding="latin1")
    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    # features except "headline" ("up", "stay", "down") are viewed as labels (three classes)
    labels = [label for label in dataset['train'].features.keys() if label not in ["headline"]]
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}

    # tokenize headline in each record
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

    # pretrained BERT model for classification
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                               problem_type="multi_label_classification", 
                                                               num_labels=len(labels),
                                                               id2label=id2label,
                                                               label2id=label2id)
    
    # training configuration
    batch_size = 16
    metric_name = "f1"
    args = TrainingArguments(
        f"bert-finetuned-sem_eval-english",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
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
    train_output = trainer.train()
    test_output = trainer.evaluate()
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("training results:\n")
        f.write(str(train_output) + "\n\n")
        f.write("testing results:\n")
        f.write(str(test_output) + "\n")