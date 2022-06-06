import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml_project.model.train_params import model_params

from ml_project.data_processing.processing import preprocess_data

DATA_PATH = '~/Downloads/heart_cleveland_upload.csv'
MODEL_PICKLE_PATH = './model/finalized_model.sav'


def train():
    mp = model_params()
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=mp.train_test_split)

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    train_y, test_y = train_df.iloc[:, -1].values, test_df.iloc[:, -1].values
    train_x, test_x = train_df.iloc[:, :-1].values, test_df.iloc[:, :-1].values

    model = LogisticRegression(penalty=mp.penalty, random_state=mp.random_state)
    model.fit(train_x, train_y)

    pickle.dump(model, open(MODEL_PICKLE_PATH, 'wb'))
    return


def predict(model, data, columns):
    df = preprocess_data(pd.DataFrame(data=data, columns=columns))

    predicted_y = model.predict(df.values)
    return predicted_y



