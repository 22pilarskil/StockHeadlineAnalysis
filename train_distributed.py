import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, AdamW
from data_loader import StockHeadlineDataset
from sklearn.metrics import accuracy_score, f1_score
from util import *
import argparse
import os
from model import StockPredictor
from pathlib import Path
import time

BASE_PRINT_EVERY = 1000  # Base frequency for single-GPU case
EARLY_EXIT = None

def get_print_every(world_size):
    """Adjust print frequency based on number of GPUs."""
    return max(1, BASE_PRINT_EVERY // world_size)  # Ensure at least 1

def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'  # Single node; adjust for multi-node
    os.environ['MASTER_PORT'] = '12345'      # Arbitrary port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # NCCL for GPU

def cleanup_distributed():
    """Destroy the distributed environment."""
    dist.destroy_process_group()

def train_epoch(model, data_loader, loss_function, optimizer, device, epoch, neutral_window, rank=0, world_size=1):
    model.train()
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_labels = []
    skipped_nan = 0
    start = time.time()

    PRINT_EVERY = get_print_every(world_size)  # Adjusted per-GPU frequency

    for batch_num, batch in enumerate(data_loader):
        input_ids = batch['headline_input_ids'].to(device)
        attention_mask = batch['headline_attention_mask'].to(device)
        financial_data = batch['features'].to(device)
        labels = batch['label'].to(device)
        labels = convert_to_class_indices(labels, neutral_window)

        nan_mask = ~torch.isnan(financial_data).any(dim=-1).any(dim=-1)
        if not nan_mask.all():
            input_ids = input_ids[nan_mask]
            attention_mask = attention_mask[nan_mask]
            financial_data = financial_data[nan_mask]
            labels = labels[nan_mask]
            skipped_nan += (data_loader.batch_size - nan_mask.sum())
            if nan_mask.sum() == 0:
                continue

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask, financial_data=financial_data)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(true_labels)

        if batch_num % PRINT_EVERY == 0 and batch_num > 0:
            # Synchronize loss across all GPUs
            loss_tensor = torch.tensor(total_loss / total_batches if total_batches > 0 else 0, device=device)
            if world_size > 1:
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss_tensor /= world_size  # Average across GPUs
            avg_loss = loss_tensor.item()

            if rank == 0:
                global_batch_num = batch_num * world_size
                total_batches_global = len(data_loader) * world_size
                print("EPOCH {}: BATCH {}/{}, LOSS: {:.4f}, skipped: {}, time taken: {:.2f}".format(
                    epoch, global_batch_num, total_batches_global, avg_loss, skipped_nan, time.time() - start))
            total_loss = 0  # Reset after printing
            total_batches = 0
            start = time.time()

        if EARLY_EXIT is not None and batch_num > EARLY_EXIT:
            break

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    if world_size > 1:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def evaluate(model, data_loader, loss_function, device, epoch, neutral_window, batch_size, rank=0, world_size=1):
    model.eval()
    total_loss = 0
    total_batches = 0
    all_preds = []
    all_labels = []
    positive_samples = 0
    negative_samples = 0
    neutral_samples = 0

    PRINT_EVERY = get_print_every(world_size)  # Adjusted per-GPU frequency
    start = time.time()

    with torch.no_grad():
        for batch_num, batch in enumerate(data_loader):
            input_ids = batch['headline_input_ids'].to(device)
            attention_mask = batch['headline_attention_mask'].to(device)
            financial_data = batch['features'].to(device)
            labels = batch['label'].to(device)
            labels = convert_to_class_indices(labels, neutral_window)

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

            logits = model(input_ids, attention_mask, financial_data)
            loss = loss_function(logits, labels)

            total_loss += loss.item()
            total_batches += 1
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true_labels = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(true_labels)

            if batch_num % PRINT_EVERY == 0 and batch_num > 0:
                loss_tensor = torch.tensor(total_loss / total_batches if total_batches > 0 else 0, device=device)
                if world_size > 1:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    loss_tensor /= world_size
                avg_loss = loss_tensor.item()

                if rank == 0:
                    global_batch_num = batch_num * world_size
                    total_batches_global = len(data_loader) * world_size
                    print("EPOCH {}: BATCH {}/{}, LOSS: {:.4f}, time taken: {:.2f}".format(
                        epoch, global_batch_num, total_batches_global, avg_loss, time.time() - start))
                total_loss = 0
                total_batches = 0
                start = time.time()

            if EARLY_EXIT is not None and batch_num > EARLY_EXIT:
                break

    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    if world_size > 1:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Synchronize sample counts and metrics for reporting
    if world_size > 1:
        counts = torch.tensor([negative_samples, neutral_samples, positive_samples], device=device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        negative_samples, neutral_samples, positive_samples = counts.tolist()
        metrics = torch.tensor([accuracy, f1], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        accuracy, f1 = metrics.tolist()
        accuracy /= world_size
        f1 /= world_size

    if rank == 0:  # Only rank 0 prints summary
        print("EPOCH {}:\nAverage Loss: {}\nAccuracy: {}\nF1 Score: {}".format(epoch, avg_loss, accuracy, f1))
        print("Data distribution:\nNegative Samples: {}\nNeutral Samples: {}\nPositive Samples: {}".format(
            negative_samples, neutral_samples, positive_samples))
        print("Negative %: {:.4f}\nNeutral %: {:.4f}\nPositive %: {:.4f}".format(
            negative_samples / (total_batches * batch_size * world_size), 
            neutral_samples / (total_batches * batch_size * world_size),
            positive_samples / (total_batches * batch_size * world_size)))
        print("-----------------------------------\n")

    return avg_loss, accuracy, f1

def train(rank, world_size, args):
    """Main training function for distributed or single process."""
    if args.distributed:
        setup_distributed(rank, world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset setup
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = StockHeadlineDataset(
        headline_file=args.headline_file,
        stock_data_dir=args.stock_data_dir,
        tokenizer=tokenizer,
        n_samples=None,
        lookback_days=14,
        future_days=1,
        cache_size=50,
        chunk_size=10000,
        device=device,
        verbose=True,
        find_valid_samples=False
    )

    dataset_size = len(dataset)
    test_ratio = 0.2
    train_size = int((1 - test_ratio) * dataset_size)
    test_size = dataset_size - train_size

    torch.manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # DataLoader with DistributedSampler if distributed
    num_workers = min(8, os.cpu_count())
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    # Model setup
    model = StockPredictor(num_features=11).to(device)
    if args.distributed:
        model = DDP(model, device_ids=[rank])  # Wrap with DDP
    # torch.compile(model)  # Uncomment if PyTorch 2.0+ and desired

    class_weights = torch.tensor([0.9142, 1.2945, 0.8482], dtype=torch.float).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Load checkpoint if exists
    epoch = 0
    checkpoint_path = args.checkpoint_path if not args.distributed or rank == 0 else f"{args.checkpoint_path}_{rank}"
    if os.path.exists(checkpoint_path):
        model, optimizer, epoch, neutral_window = load_model(checkpoint_path, model, optimizer)
    elif rank == 0:  # Only rank 0 evaluates and saves initially
        print("Evaluating at epoch 0")
        create_report_file(args.report_path, args.batch_size, args.neutral_window)
        avg_loss, accuracy, f1 = evaluate(model, test_dataloader, loss_fn, device, epoch, args.neutral_window, args.batch_size, rank, world_size)
        append_loss_data(args.report_path, epoch, avg_loss, accuracy, f1)

    # Training loop
    if rank == 0:
        print("Training")
    for i in range(epoch, args.epochs):
        epoch += 1
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)  # Ensure shuffling per epoch
        train_loss, train_acc = train_epoch(model, train_dataloader, loss_fn, optimizer, device, epoch, args.neutral_window, rank, world_size)
        avg_loss, accuracy, f1 = evaluate(model, test_dataloader, loss_fn, device, epoch, args.neutral_window, args.batch_size, rank, world_size)

        # Metrics are already synchronized in evaluate; only rank 0 writes
        if rank == 0:
            append_loss_data(args.report_path, epoch, avg_loss, accuracy, f1)
            save_model(epoch, model, optimizer, args.neutral_window, checkpoint_path)

    if args.distributed:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Stock Prediction Training")
    parser.add_argument('--distributed', action='store_true', help="Enable distributed training")
    parser.add_argument('--headline_file', type=str, default="data/headline/filtered_headlines_2015_error_removed.csv")
    parser.add_argument('--stock_data_dir', type=str, default="data/stock/processed")
    parser.add_argument('--checkpoint_path', type=str, default="/scratch/model_checkpoint.pth")
    parser.add_argument('--report_path', type=str, default="report.txt")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--neutral_window', type=float, default=0.005)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    if args.distributed:
        world_size = torch.cuda.device_count()  # Number of GPUs
        print("WORKING WITH {} GPUS".format(world_size))
        mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train(0, 1, args)  # Single process, rank=0, world_size=1

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()