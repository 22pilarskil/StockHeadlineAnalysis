### 1. prepare data

Create a folder names `data`. Put headlines file `filtered_headlines_2015_error_removed.csv` under `data/headline/`. Put numerical data files under `data/stock/processed/`.

### 2. preprocess data

Run `find_nans.py` and `get_scalers.py`.

### 3. train model

When first train model, add `--find_valid_samples` as commandline argument to `train.py` to generate a file named `valid_samples.pkl`.
