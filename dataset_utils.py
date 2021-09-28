__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def train_test_df_split(file_path, target_feature,
                     rand_seed=42, test_size=0.2):

    df = pd.read_csv(file_path)
    X, y = df[df.columns.difference([target_feature])], df[target_feature]
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y,
                                                        random_state=rand_seed)

    df_train = pd.merge(X_train, y_train, left_index=True, right_index=True)
    df_test = pd.merge(X_test, y_test, left_index=True, right_index=True)
    dir_name, file_name = os.path.dirname(file_path), os.path.basename(file_path).split(".")[0]
    train_file = os.path.join(dir_name, file_name + "_train.csv")
    test_file = os.path.join(dir_name, file_name + "_test.csv")
    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)

    return train_file, test_file



