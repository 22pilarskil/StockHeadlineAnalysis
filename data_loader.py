import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import OrderedDict


class StockHeadlineDataset(Dataset):
    def __init__(
        self,
        headline_file: str,
        stock_data_dir: str,
        n_samples: int = None,
        lookback_days: int = 14,
        future_days: int = 1,
        price_movement_threshold: float = 0.005,
        cache_size: int = 50,
        chunk_size: int = 10000,
        device=None,
        verbose: bool = False,
    ):
        """
        Dataset for stock headlines and corresponding stock data

        Args:
            headline_file (str): Path to the CSV file with headlines
            stock_data_dir (str): Directory containing stock data CSV files
            n_samples (int, optional): Number of headline samples to use. If None, use all.
            lookback_days (int): Number of days to look back for stock features
            future_days (int): Number of days to look forward for price change calculation
            price_movement_threshold (float): Threshold for determining price movement direction
            cache_size (int): Maximum number of stock dataframes to keep in memory
            chunk_size (int): Number of headlines to process at once. If None, process all at once.
            device: Device to place tensors on ('cuda' or 'cpu')
            verbose (bool): Whether to print detailed progress information
        """
        # Store parameters but don't load all headlines at once
        self.headline_file = headline_file
        self.stock_data_dir = Path(stock_data_dir)
        self.n_samples = n_samples
        self.chunk_size = chunk_size
        self.verbose = verbose

        # Get the total number of rows in the headline file
        with open(headline_file, "rb") as f:
            self.total_rows = sum(1 for _ in f) - 1  # Subtract 1 for header

        if n_samples is not None:
            self.total_rows = min(self.total_rows, n_samples)

        self.lookback_days = lookback_days
        self.future_days = future_days
        self.price_movement_threshold = price_movement_threshold

        # Create a cache for stock data with limited size
        self.cache_size = cache_size
        self.stock_data_cache = OrderedDict()

        # Create a mapping of available stock tickers
        self.available_tickers = set()
        for stock_file in self.stock_data_dir.glob("*.csv"):
            self.available_tickers.add(stock_file.stem)

        if self.verbose:
            print(f"Found {len(self.available_tickers)} stock tickers in directory")

        # Set device for tensor placement
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.verbose:
            print(f"Using device: {self.device}")

        # Store headline data only for valid samples
        self.headline_data = {}  # Dictionary to store headline data for valid samples
        self.valid_samples = []
        self._find_valid_samples()

    def _load_stock_data(self, ticker: str) -> pd.DataFrame:
        """Load stock data for a specific ticker with LRU caching"""
        # Return from cache if available
        if ticker in self.stock_data_cache:
            # Move to the end (most recently used)
            data = self.stock_data_cache.pop(ticker)
            self.stock_data_cache[ticker] = data
            return data

        # Load from file
        try:
            stock_file = self.stock_data_dir / f"{ticker}.csv"
            data = pd.read_csv(stock_file)
            data["Date"] = pd.to_datetime(data["Date"])
            data = data.sort_values("Date")

            # Add to cache
            self.stock_data_cache[ticker] = data

            # Remove oldest item if cache is full
            if len(self.stock_data_cache) > self.cache_size:
                self.stock_data_cache.popitem(last=False)

            return data
        except Exception as e:
            print(f"Error loading {ticker}.csv: {e}")
            return None

    def _find_valid_samples(self):
        """Find headlines with corresponding valid stock data for analysis"""
        if self.verbose:
            print(f"Starting validation of up to {self.total_rows} headlines")

        missing_tickers = 0
        missing_date_data = 0
        insufficient_history = 0
        insufficient_future = 0
        total_processed = 0

        # Determine how to read the CSV file
        if self.chunk_size is None:
            if self.n_samples is not None:
                if self.verbose:
                    print(f"Loading first {self.n_samples} headlines at once")
                chunks = pd.read_csv(self.headline_file, nrows=self.n_samples)
            else:
                if self.verbose:
                    print("Loading entire headline file at once")
                chunks = pd.read_csv(self.headline_file)

            # Convert a single DataFrame into a list with one item for consistent processing
            chunks = [chunks]
        else:
            # Process the headline file in chunks to reduce memory usage
            if self.verbose:
                print(f"Processing headline file in chunks of size {self.chunk_size}")
            chunks = pd.read_csv(self.headline_file, chunksize=self.chunk_size)

        for chunk_num, chunk in enumerate(chunks):
            # Terminate if we've processed enough samples when using chunks
            if (
                self.chunk_size is not None
                and self.n_samples is not None
                and total_processed >= self.n_samples
            ):
                break

            # Trim chunk if needed to respect n_samples limit
            if self.n_samples is not None and self.chunk_size is not None:
                remaining = self.n_samples - total_processed
                if remaining < len(chunk):
                    chunk = chunk.iloc[:remaining]

            if self.verbose:
                print(
                    f"Processing chunk {chunk_num + 1}, headlines {total_processed + 1}-{total_processed + len(chunk)}"
                )

            # Group headlines by ticker to minimize file loading during validation
            headlines_by_ticker = {}
            for idx, row in chunk.iterrows():
                ticker = row["Stock_symbol"]
                if ticker not in headlines_by_ticker:
                    headlines_by_ticker[ticker] = []
                headlines_by_ticker[ticker].append((idx, row))

            # Process each ticker group
            for ticker, headline_entries in headlines_by_ticker.items():
                # Skip if we don't have this ticker's data file
                if ticker not in self.available_tickers:
                    missing_tickers += len(headline_entries)
                    continue

                # Load this ticker's data once for validating all its headlines
                stock_df = self._load_stock_data(ticker)
                if stock_df is None:
                    missing_tickers += len(headline_entries)
                    continue

                for idx, row in headline_entries:
                    headline_date = pd.to_datetime(row["Date"])

                    # Find the exact trading date matching the headline date
                    headline_date_only = headline_date.date()
                    mask = stock_df["Date"].dt.date == headline_date_only
                    relevant_dates = stock_df[mask]["Date"]
                    if relevant_dates.empty:
                        missing_date_data += 1
                        continue

                    headline_trading_date = relevant_dates.iloc[0]
                    headline_idx = stock_df[
                        stock_df["Date"] == headline_trading_date
                    ].index[0]

                    # Check if we have enough historical data (lookback days)
                    if headline_idx < self.lookback_days:
                        insufficient_history += 1
                        continue

                    # Check if we have enough future data (future days)
                    future_dates = stock_df[stock_df["Date"] > headline_trading_date]
                    if len(future_dates) < self.future_days:
                        insufficient_future += 1
                        continue

                    future_day_idx = future_dates.index[self.future_days - 1]

                    # Store only the valid headline data and sample info
                    self.headline_data[idx] = row
                    self.valid_samples.append(
                        (idx, ticker, headline_idx, future_day_idx)
                    )

            total_processed += len(chunk)

            # Break after the single chunk if we loaded all at once
            if self.chunk_size is None:
                break

            # Early termination if we've found enough samples
            if self.n_samples is not None and len(self.valid_samples) >= self.n_samples:
                if self.verbose:
                    print(f"Reached requested sample count: {self.n_samples}")
                break

        # Clear cache after validation to free memory
        self.stock_data_cache.clear()

        # Print debugging information
        if self.verbose:
            print(f"Missing tickers: {missing_tickers}")
            print(f"Missing date data: {missing_date_data}")
            print(f"Insufficient history: {insufficient_history}")
            print(f"Insufficient future data: {insufficient_future}")
            print(f"Valid samples found: {len(self.valid_samples)}")

    def __len__(self):
        """Return the number of valid samples"""
        return len(self.valid_samples)

    def __getitem__(self, idx: int):
        """Get a data sample: historical prices, headline, and label"""
        headline_idx, ticker, current_idx, future_day_idx = self.valid_samples[idx]

        # Get headline information from stored valid headline data
        headline_row = self.headline_data[headline_idx]
        headline_text: str = headline_row["Article_title"]

        # Load stock data on demand
        stock_df = self._load_stock_data(ticker)

        # Get historical data (lookback period)
        historical_data = stock_df.iloc[
            current_idx - self.lookback_days : current_idx + 1
        ]

        # Extract features for the lookback period
        features = historical_data[["Open", "High", "Low", "Close", "Volume"]].values
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        # Calculate price change for labeling
        current_close = stock_df.iloc[current_idx]["Close"]
        future_close = stock_df.iloc[future_day_idx]["Close"]

        price_change_ratio = (future_close - current_close) / current_close

        # Assign label based on price movement rules using the threshold parameter
        if price_change_ratio > self.price_movement_threshold:
            label = 2  # Increase
        elif price_change_ratio < -self.price_movement_threshold:
            label = 0  # Decrease
        else:
            label = 1  # Stayed the same

        return {
            "headline": headline_text,
            "ticker": ticker,
            "features": features_tensor,
            "label": torch.tensor(label, dtype=torch.int8).to(self.device),
            "price_change_ratio": torch.tensor(
                price_change_ratio, dtype=torch.float32
            ).to(self.device),
        }


def create_stock_data_loader(
    headline_file: str,
    stock_data_dir: str,
    n_samples: int = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    lookback_days: int = 14,
    future_days: int = 1,
    price_movement_threshold: float = 0.005,
    cache_size: int = 50,
    chunk_size: int = 10000,
    device=None,
    verbose: bool = False,
) -> DataLoader:
    """
    Creates a DataLoader for stock headline data

    Args:
        headline_file (str): Path to the headline CSV file
        stock_data_dir (str): Directory containing stock CSV files
        n_samples (int, optional): Number of headline samples to use. If None, use all.
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for the DataLoader
        lookback_days (int): Number of days to look back for stock features
        future_days (int): Number of days to look forward for price change calculation
        price_movement_threshold (float): Threshold for determining price movement direction
        cache_size (int): Maximum number of stock dataframes to keep in memory
        chunk_size (int): Number of headlines to process at once
        device: Device to place tensors on ('cuda' or 'cpu')
        verbose (bool): Whether to print detailed progress information

    Returns:
        DataLoader: A PyTorch DataLoader for the stock headline dataset
    """
    dataset = StockHeadlineDataset(
        headline_file=headline_file,
        stock_data_dir=stock_data_dir,
        n_samples=n_samples,
        lookback_days=lookback_days,
        future_days=future_days,
        price_movement_threshold=price_movement_threshold,
        cache_size=cache_size,
        chunk_size=chunk_size,
        device=device,
        verbose=verbose,  # Pass verbose parameter to dataset
    )

    valid_samples = len(dataset)
    if verbose:
        print(
            f"Found {valid_samples} valid samples out of approximately {dataset.total_rows} headlines"
        )

    # Handle case with zero valid samples
    if valid_samples == 0:
        raise ValueError(
            "No valid samples found. Please check your data paths, file formats, "
            "and matching criteria (lookback_days and future_days)."
        )

    # Adjust batch size if needed to ensure it's not larger than dataset
    actual_batch_size = min(batch_size, valid_samples)
    if actual_batch_size != batch_size and verbose:
        print(
            f"Adjusted batch size from {batch_size} to {actual_batch_size} due to dataset size"
        )

    loader = DataLoader(
        dataset, batch_size=actual_batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return loader


if __name__ == "__main__":
    headline_file = "data/headline/filtered_headlines_2015_error_removed.csv"
    stock_data_dir = "data/stock/raw"

    headline_path = Path(headline_file)
    stock_dir_path = Path(stock_data_dir)

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

    try:
        loader = create_stock_data_loader(
            headline_file=headline_file,
            stock_data_dir=stock_data_dir,
            n_samples=None,  # Use None for all headlines
            batch_size=16,
            lookback_days=14,
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
