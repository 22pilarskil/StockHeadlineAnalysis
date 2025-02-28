import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class StockHeadlineDataset(Dataset):
    def __init__(
        self,
        headline_file: str,
        stock_data_dir: str,
        n_samples: int = None,
        lookback_days: int = 14,
        future_days: int = 1,
        device=None,
    ):
        """
        Dataset for stock headlines and corresponding stock data

        Args:
            headline_file (str): Path to the CSV file with headlines
            stock_data_dir (str): Directory containing stock data CSV files
            n_samples (int, optional): Number of headline samples to use. If None, use all.
            lookback_days (int): Number of days to look back for stock features
            future_days (int): Number of days to look forward for price change calculation
            device: Device to place tensors on ('cuda' or 'cpu')
        """
        # Load headline data
        self.stock_data_dir = Path(stock_data_dir)
        self.headlines = pd.read_csv(headline_file)
        if n_samples is not None:
            self.headlines = self.headlines.head(n_samples)

        self.lookback_days = lookback_days
        self.future_days = future_days

        # Set device for tensor placement
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        # Pre-load all stock data files
        self.stock_data = {}
        for stock_file in self.stock_data_dir.glob("*.csv"):
            ticker = stock_file.stem
            try:
                data = pd.read_csv(stock_file)
                data["Date"] = pd.to_datetime(data["Date"])
                data = data.sort_values("Date")
                self.stock_data[ticker] = data
            except Exception as e:
                print(f"Error loading {stock_file}: {e}")

        print(f"Loaded stock data for {len(self.stock_data)} tickers")

        # Find valid samples
        self.valid_samples = []
        self._find_valid_samples()

    def _find_valid_samples(self):
        """Find headlines with corresponding valid stock data for analysis"""
        print(f"Starting with {len(self.headlines)} headlines to analyze")

        missing_tickers = 0
        missing_date_data = 0
        insufficient_history = 0
        insufficient_future = 0

        for idx, row in self.headlines.iterrows():
            ticker = row["Stock_symbol"]
            headline_date: pd.Timestamp = pd.to_datetime(row["Date"])

            # Skip if we don't have data for this ticker
            if ticker not in self.stock_data:
                missing_tickers += 1
                continue

            stock_df: pd.DataFrame = self.stock_data[ticker]

            # Find the exact trading date matching the headline date
            headline_date_only = headline_date.date()
            mask = stock_df["Date"].dt.date == headline_date_only
            relevant_dates = stock_df[mask]["Date"]
            if relevant_dates.empty:
                missing_date_data += 1
                continue

            headline_trading_date = relevant_dates.iloc[0]
            headline_idx = stock_df[stock_df["Date"] == headline_trading_date].index[0]

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

            # Store valid sample: headline index, current day index, future day index
            self.valid_samples.append((idx, headline_idx, future_day_idx))

        # Print debugging information
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
        headline_idx, current_idx, future_day_idx = self.valid_samples[idx]

        # Get headline information
        headline_row = self.headlines.iloc[headline_idx]
        ticker: str = headline_row["Stock_symbol"]
        headline_text: str = headline_row["Article_title"]

        # Get stock data
        stock_df: pd.DataFrame = self.stock_data[ticker]

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

        # Assign label based on price movement rules
        if price_change_ratio > 0.005:
            label = 1  # Increase
        elif price_change_ratio < -0.005:
            label = -1  # Decrease
        else:
            label = 0  # Stayed the same

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
    device=None,
) -> DataLoader:
    """
    Creates a DataLoader for stock headline data

    Args:
        headline_file (str): Path to the headline CSV file
        stock_data_dir (str): Directory containing stock CSV files
        n_samples (int, optional): Number of samples to use
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of workers for the DataLoader
        lookback_days (int): Number of days to look back for stock features
        future_days (int): Number of days to look forward for price change calculation
        device: Device to place tensors on ('cuda' or 'cpu')

    Returns:
        DataLoader: A PyTorch DataLoader for the stock headline dataset
    """
    dataset = StockHeadlineDataset(
        headline_file=headline_file,
        stock_data_dir=stock_data_dir,
        n_samples=n_samples,
        lookback_days=lookback_days,
        future_days=future_days,
        device=device,
    )

    valid_samples = len(dataset)
    print(
        f"Found {valid_samples} valid samples out of {len(dataset.headlines)} headlines"
    )

    # Handle case with zero valid samples
    if valid_samples == 0:
        raise ValueError(
            "No valid samples found. Please check your data paths, file formats, "
            "and matching criteria (lookback_days and future_days)."
        )

    # Adjust batch size if needed to ensure it's not larger than dataset
    actual_batch_size = min(batch_size, valid_samples)
    if actual_batch_size != batch_size:
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
            n_samples=10000,  # Use None for all samples
            batch_size=16,
            lookback_days=14,
            future_days=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
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
