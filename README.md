# tmlcc2021

## Requirements
install python libraries:
```
pip install -r requirements.txt
```

## Usage (main.py)
```
python main.py -r [rep1] [rep2] ... [repn] -m [model_name] -d [directory]

# Enable grid search
python main.py -r [rep1] [rep2] ... [repn] -m [model_name] -d results --grid_search

# Example
python main.py -r preprocessed linearAP-RDF -m rf -d results --grid_search

# Example (Prediction)
python main.py -r preprocessed linearAP-RDF -m rf -d results --mode test
```
*Note: `rep` should follow the representation's folder name.*

## Usage (ensemble.py)
```
python ensemble.py -d [directory] -p [csv_file1] [csv_file2] ... [csv_filen]

# Example
python ensemble.py -d results -p preprocessed_rf.csv preprocessed_gbr.csv
```

## Usage (opt.py)
Run hyperparameter search with Optuna.
Only `CatBoostRegressor` is supported for now.
```
python opt.py -r [rep1] [rep2] ... [repn] -m cat -d [directory] -n [n_trials]

# Example
python opt.py -r preprocessed linearAP-RDF -m cat -d results -n 10
```