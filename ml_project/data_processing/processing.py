import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    num_feat = ['age', 'trestbps', 'oldpeak', 'thalach', 'chol']
    for col in num_feat:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].apply(lambda x: (x - mean) / (std ** 2))

    cat_feat_possible_values = {
        'sex': [0, 1],
        'cp': [0, 1, 2, 3],
        'fbs': [0, 1],
        'restecg': [0, 1, 2],
        'exang': [0, 1],
        'slope': [0, 1, 2],
        'ca': [0, 1, 2, 3],
        'thal': [0, 1, 2],
    }
    for cf in cat_feat_possible_values.keys():
        for pv in cat_feat_possible_values[cf]:
            df[f'{cf}_{pv}'] = (df[cf] == pv).astype('int')
        df.drop(cf, axis=1, inplace=True)

    return df