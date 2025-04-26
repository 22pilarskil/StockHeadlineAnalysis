import torch
from transformers import BertTokenizer, AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from data_loader import StockHeadlineDataset
from sklearn.metrics import accuracy_score, f1_score
from util import *
import argparse
import os
from model import StockPredictor
import time
import multiprocessing as mp
import random
import numpy as np
from pathlib import Path


def train_epoch(model, data_loader, loss_function, optimizer, device, epoch, args, scaler):
    model.train()
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_labels = []
    skipped_nan = 0
    start = time.time()

    for batch_num, batch in enumerate(data_loader):
        input_ids = batch['headline_input_ids'].to(device)
        attention_mask = batch['headline_attention_mask'].to(device)
        financial_data = batch['features'].to(device)
        labels = batch['label'].to(device)
        labels = convert_to_class_indices(labels, args.neutral_window)

        nan_mask = ~torch.isnan(financial_data).any(dim=-1).any(dim=-1)
        if not nan_mask.all():
            input_ids = input_ids[nan_mask]
            attention_mask = attention_mask[nan_mask]
            financial_data = financial_data[nan_mask]
            labels = labels[nan_mask]
            skipped_nan += (args.batch_size - nan_mask.sum())
            if nan_mask.sum() == 0:
                continue

        optimizer.zero_grad()
        with autocast():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)
            loss = loss_function(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_num % args.print_every == 0:
            print(f"EPOCH {epoch}: BATCH {batch_num}/{len(data_loader)}, LOSS: {loss.item():.4f}, skipped: {skipped_nan}, time taken: {time.time() - start:.2f}")
            start = time.time()

        total_loss += loss.item()
        total_batches += 1
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(true_labels)

        if args.early_exit is not None and batch_num > args.early_exit:
            break

    avg_loss = total_loss / total_batches
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def evaluate(model, data_loader, loss_function, device, epoch, args):
    model.eval()
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_labels = []
    positive_samples = 0
    negative_samples = 0
    neutral_samples = 0
    start = time.time()

    with torch.no_grad():
        for batch_num, batch in enumerate(data_loader):
            input_ids = batch['headline_input_ids'].to(device)
            attention_mask = batch['headline_attention_mask'].to(device)
            financial_data = batch['features'].to(device)
            labels = batch['label'].to(device)
            labels = convert_to_class_indices(labels, args.neutral_window)

            nan_mask = ~torch.isnan(financial_data).any(dim=-1).any(dim=-1)
            if not nan_mask.all():
                input_ids = input_ids[nan_mask]
                attention_mask = attention_mask[nan_mask]
                financial_data = financial_data[nan_mask]
                labels = labels[nan_mask]
                if nan_mask.sum() == 0:
                    continue

            counts = torch.bincount(labels, minlength=3)
            negative_samples += counts[0]
            neutral_samples += counts[1]
            positive_samples += counts[2]

            with autocast():
                logits = model(input_ids, attention_mask, financial_data)
                loss = loss_function(logits, labels)

            total_loss += loss.item()
            total_batches += 1
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(true_labels)

            if batch_num % args.print_every == 0:
                print(f"EPOCH {epoch}: BATCH {batch_num}/{len(data_loader)}, LOSS: {loss.item():.4f}, time taken: {time.time() - start:.2f}")
                start = time.time()

            if args.early_exit is not None and batch_num > args.early_exit:
                break

    avg_loss = total_loss / total_batches
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"EPOCH {epoch}:\nAverage Loss: {avg_loss}\nAccuracy: {accuracy}\nF1 Score: {f1}")
    print(f"Data distribution:\nNegative Samples: {negative_samples}\nNeutral Samples: {neutral_samples}\nPositive Samples: {positive_samples}")
    print(f"Negative %: {negative_samples / total_batches / args.batch_size:.4f}\nNeutral %: {neutral_samples / total_batches / args.batch_size:.4f}\nPositive %: {positive_samples / total_batches / args.batch_size:.4f}")
    print("-----------------------------------\n")

    return avg_loss, accuracy, f1

def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    mp.set_start_method('spawn', force=True)

    scaler = GradScaler()  # Moved from global scope

    headline_path = Path(args.headline_file)
    stock_dir_path = Path(args.stock_data_dir)

    if not headline_path.exists():
        print(f"ERROR: Headline file not found: {headline_path.absolute()}")
    else:
        print(f"Found headline file: {headline_path.absolute()}")

    if not stock_dir_path.exists():
        print(f"ERROR: Stock data directory not found: {stock_dir_path.absolute()}")
    else:
        print(f"Found stock directory: {stock_dir_path.absolute()}")
        stock_files = list(stock_dir_path.glob("*.csv"))
        print(f"Stock directory contains {len(stock_files)} CSV files")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StockHeadlineDataset(
        headline_file=args.headline_file,
        stock_data_dir=args.stock_data_dir,
        tokenizer=tokenizer,
        n_samples=args.n_samples,
        lookback_days=args.lookback_days,
        future_days=args.future_days,
        cache_size=args.cache_size,
        chunk_size=args.chunk_size,
        device=device,
        verbose=args.verbose,
        find_valid_samples=args.find_valid_samples
    )

    dataset_size = len(dataset)
    train_size = int((1 - args.test_ratio) * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    num_workers = min(args.max_workers, os.cpu_count())
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    model = StockPredictor(num_features=args.num_features)
    if args.compile_model:
        torch.compile(model)

    class_weights = torch.tensor(args.class_weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    model = model.to(device)

    epoch = 0
    if os.path.exists(args.checkpoint_path):
        model, optimizer, epoch, neutral_window = load_model(args.checkpoint_path, model, optimizer)
        args.neutral_window = neutral_window
    else:
        print("Evaluating at epoch 0")
        create_report_file(args.report_path, args.batch_size, args.neutral_window)
        avg_loss, accuracy, f1 = evaluate(model, test_dataloader, loss_fn, device, epoch, args)
        append_loss_data(args.report_path, epoch, avg_loss, accuracy, f1)

    print("Training")
    for i in range(epoch, args.epochs):
        epoch += 1
        train_epoch(model, train_dataloader, loss_fn, optimizer, device, epoch, args, scaler)
        avg_loss, accuracy, f1 = evaluate(model, test_dataloader, loss_fn, device, epoch, args)
        append_loss_data(args.report_path, epoch, avg_loss, accuracy, f1)
    save_model(epoch, model, optimizer, args.neutral_window, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Prediction Training")
    parser.add_argument('--headline_file', type=str, default="data/headline/filtered_headlines_2015_error_removed.csv")
    parser.add_argument('--stock_data_dir', type=str, default="data/stock/processed")
    parser.add_argument('--checkpoint_path', type=str, default="/scratch/model_checkpoint.pth")
    parser.add_argument('--report_path', type=str, default="report.txt")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--neutral_window', type=float, default=0.005)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--print_every', type=int, default=1000)
    parser.add_argument('--early_exit', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--lookback_days', type=int, default=14)
    parser.add_argument('--future_days', type=int, default=1)
    parser.add_argument('--cache_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=10000)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--find_valid_samples', action='store_true', default=False)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--num_features', type=int, default=11)
    parser.add_argument('--class_weights', type=float, nargs=3, default=[0.9142, 1.2945, 0.8482])
    parser.add_argument('--compile_model', action='store_true', default=False)
    args = parser.parse_args()

    main(args)