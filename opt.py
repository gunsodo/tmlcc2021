import argparse
import joblib
import os
import pandas as pd
import optuna

from representation.pipeline import load_representation
from runner import train, predict, objective

parser = argparse.ArgumentParser("Experimental Test Run (Optuna)")

parser.add_argument('-r', '--reps', nargs='+', default=["preprocessed"])
parser.add_argument('-m', '--model', type=str, default="svr")
parser.add_argument('-n', '--n_trials', type=int, default=10)
parser.add_argument('-d', '--directory', type=str, required=True)
parser.add_argument('--mode', type=str, default="train")

args = parser.parse_args()

def print_settings(args):
    print("------------------------------")
    print("Experimental Test Run (Optuna)")
    print("------------------------------")
    print("Representations:")
    
    for rep in args.reps:
        print(f"\t{rep}")
    
    print(f"\nModel:")
    print(f"\t{args.model}")

    print(f"\nNumber of trials:")
    print(f"\t{args.n_trials}")

    print("\nMode:")
    print(f"\t{'Prediction mode' if args.mode != 'train' else 'Training mode'}")

    print("\nModel directory:")
    print(f"\t{args.directory}")
    print("------------------------------")

def main(args):
    print_settings(args)
    reps = load_representation(args.reps, args.mode)
    filename = args.directory + "/" + "_".join(["optuna"] + sorted(args.reps)) + "_" + args.model + ".sav"

    if not os.path.exists(args.directory):
            os.makedirs(args.directory)

    if args.mode != "train":
        loaded_model = joblib.load(filename)

        df, X_test = reps
        df = df.rename(columns={"MOFname": "id"})
        df["id"] = df["id"].apply(lambda x: x[9:])
        print("Predicting...")
        y_pred = loaded_model.predict(X_test)
    
        df['CO2_working_capacity [mL/g]'] = y_pred
        filename = filename[:-4] + ".csv"
        df.to_csv(filename, sep=',', index=False)
        
    else:
        X_train, X_test, y_train, y_test = reps

        # OPTUNA
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, X_train, y_train, args.model), n_trials=args.n_trials)
        print('Number of finished trials:', len(study.trials))
        print('Best trial:', study.best_trial.params)

        model = train(X_train, y_train, args.model, save_dir=args.directory, param=study.best_trial.params, grid_search=False)
        lmae = predict(X_test, y_test, model)
        joblib.dump(model, filename)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)