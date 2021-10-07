import argparse
import joblib
import os
import pandas as pd

from representation.pipeline import load_representation
from runner import train, predict

parser = argparse.ArgumentParser("Experimental Test Run")

parser.add_argument('-r', '--reps', nargs='+', default=["preprocessed"])
parser.add_argument('-m', '--model', type=str, default="svr")
parser.add_argument('-d', '--directory', type=str, required=True)
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--grid_search', nargs='?', const=True, default=False)

args = parser.parse_args()

def print_settings(args):
    print("------------------------------")
    print("Experimental Test Run")
    print("------------------------------")
    print("Representations:")
    
    for rep in args.reps:
        print(f"\t{rep}")
    
    print(f"\nModel:")
    print(f"\t{args.model}")

    print("\nGrid search:")
    print(f"\t{'Yes' if args.grid_search else 'No'}")

    print("\nMode:")
    print(f"\t{'Prediction mode' if args.mode != 'train' else 'Training mode'}")

    print("\nModel directory:")
    print(f"\t{args.directory}")
    print("------------------------------")

def main(args):
    print_settings(args)
    reps = load_representation(args.reps, args.mode)
    filename = args.directory + "/" + "_".join(sorted(args.reps)) + "_" + args.model + ".sav"

    if not os.path.exists(args.directory):
            os.makedirs(args.directory)

    if args.mode != "train":
        loaded_model = joblib.load(filename)

        df, X_test = reps
        print("Predicting...")
        y_pred = loaded_model.predict(X_test)
    
        df['CO2_working_capacity [mL/g]'] = y_pred
        filename = filename[:-4] + ".csv"
        df.to_csv(filename, sep=',', index=False)
        
    else:
        X_train, X_test, y_train, y_test = reps
        model = train(X_train, y_train, args.model, args.grid_search, args.directory)
        lmae = predict(X_test, y_test, model, args.grid_search)
        joblib.dump(model, filename)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)