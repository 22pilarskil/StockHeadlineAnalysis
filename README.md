# Execution Instructions

### 1. Prepare data

Create a folder names `data`. Put headlines file `filtered_headlines_2015_error_removed.csv` under `data/headline/`. Put numerical data files under `data/stock/processed/`.

### 2. Preprocess data

Run `find_nans.py` and `get_scalers.py` to clean up NaN values in the dataset and create sklearn StandardScaler objects to scale all training data features to a normal distribution

### 3. Train model

Run the model in a slurm environment. Use the command `sbatch training.sbatch` to train the model. Note: when training the model for the first time, add `--find_valid_samples` as commandline argument to the `train.py` invocation in `training.sbatch` to generate a file named `valid_samples.pkl`. This is a final dataset preprocessing step that only needs to be invoked once. 
