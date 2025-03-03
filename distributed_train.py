from pathlib import Path
from mpi4py import MPI
import torch

from util import split_csv_by_rows_stream
from data_loader import create_stock_data_loader

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    split_csv_by_rows_stream("data/headline/filtered_headlines_2015_error_removed.csv", size, dir_path="split")
comm.Barrier()

headline_file = f"split/chunk_{rank}.csv"
stock_data_dir = "data/stock/raw"

headline_path = Path(headline_file)
stock_dir_path = Path(stock_data_dir)

try:
    loader = create_stock_data_loader(
        headline_file=headline_file,
        stock_data_dir=stock_data_dir,
        n_samples=None,  # Use None for all headlines
        batch_size=16,
        lookback_days=14,
        num_workers=1,
        future_days=1,
        price_movement_threshold=0.005,
        cache_size=50,  # Keep n most recent stock dataframes in memory
        chunk_size=10000,  # Use None for all headlines
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True,
    )

    # Display one batch
    for batch in loader:
        print(f"Batch size: {len(batch['label'])}")
        print(f"Features shape: {batch['features'].shape}")
        print(f"Labels: {batch['label']}")
        print(f"Sample headline: {batch['headline'][0]}")
        print(f"Sample ticker: {batch['ticker'][0]}")
        break
except Exception as e:
    print(f"Error creating DataLoader: {str(e)}")