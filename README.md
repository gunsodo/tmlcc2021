# tmlcc2021

## Requirements
install python libraries:
```
pip install -r requirements.txt
```

## Usage
```
python main.py -r [rep1] [rep2] ... [repn] -m [model_name] -d results

# Enable grid search
python main.py -r [rep1] [rep2] ... [repn] -m [model_name] -d results --grid_search

# Example
python main.py -r preprocessed linearAP-RDF -m rf -d results --grid_search
```
