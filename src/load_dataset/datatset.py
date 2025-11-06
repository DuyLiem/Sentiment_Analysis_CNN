import pandas as pd
import os

def load_dataset(name_file):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    #
    root_dir = os.path.abspath(os.path.join(script_dir, '../..'))

    #
    df_path = os.path.join(root_dir, 'dataset', name_file)

    print("Loading:", df_path)

    return pd.read_csv(df_path, encoding='utf-8')
