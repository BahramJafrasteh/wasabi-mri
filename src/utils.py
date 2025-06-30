import pandas as pd
import numpy as np

def get_merged(df):
    merged_columns = {}
    columns_to_drop = []

    for col in df.columns:
        if col.startswith('left '):
            right_col = col.replace('left ', 'right ')
            if right_col in df.columns:
                name = col.replace('left ', '')
                merged_columns[name] = (df[col] + df[right_col]) / 2
                columns_to_drop += [col, right_col]
        elif col.startswith('ctx-lh-'):
            right_col = col.replace('ctx-lh-', 'ctx-rh-')
            if right_col in df.columns:
                name = col.replace('ctx-lh-', '')
                merged_columns[name] = (df[col] + df[right_col]) / 2
                columns_to_drop += [col, right_col]

    df_merged = df.copy()
    df_merged = df_merged.assign(**merged_columns)
    df_merged.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    return df_merged
