import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split

def load_representation(rep_list, mode="train"):
    if len(rep_list) == 0:
        raise Exception("Error: the representation list is empty")

    dfs = []

    for rep in rep_list:
        _df = pd.read_csv(f"data/{rep}/{mode}.csv", index_col=False)
        _df['MOFname'] = _df['MOFname'].apply(lambda x: x[:-4] if ".cif" in x else x)
        dfs.append(_df)

    df_merged = reduce(lambda  left,right: pd.merge(left, right, on='MOFname', how='inner'), dfs)
    
    if mode == "train":
        gt_df = pd.read_csv("data/working_capacity/train.csv", index_col=False)
        df_merged = df_merged.merge(gt_df, how='inner', on='MOFname')
        X, y = df_merged.iloc[:, 1:-1], df_merged.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    elif mode in ["pretest", "test"]:
        df_merged.sort_values(by=['MOFname'], ).reset_index(drop=True)
        return df_merged.iloc[:, :1], df_merged.iloc[:, 1:]

    else:
        raise KeyError("model should be one of the following choices: 'train', 'pretest', 'test'")

    
