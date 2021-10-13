import argparse
import joblib
import os
import pandas as pd
import numpy as np
from uuid import uuid4

from representation.pipeline import load_representation
from runner import predict

parser = argparse.ArgumentParser("Ensemble Prediciton")

parser.add_argument('-p', '--predictions', nargs='+', default=[], required=True)
parser.add_argument('-m', '--method', type=str, default="mean")
parser.add_argument('-d', '--directory', type=str, required=True)

args = parser.parse_args()

def main(args):
    capacities = []

    for pred in args.predictions:
        _df = pd.read_csv(os.path.join(args.directory, pred), index_col=False)
        _df = _df.sort_values(by='id', ascending=True).reset_index(drop=True)
        capacities.append(np.array(_df['CO2_working_capacity [mL/g]']))
    
    capacities = np.vstack(capacities)
    mean = np.mean(capacities, axis=0)
    _df['CO2_working_capacity [mL/g]'] = mean
    _df.to_csv(os.path.join(args.directory, f'ensemble{uuid4()}.csv'), index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)