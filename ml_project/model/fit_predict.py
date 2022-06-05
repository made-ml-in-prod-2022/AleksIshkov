import sys
import logging
import logging.config
import pickle
import pandas as pd
import numpy as np

import yaml
import click
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from ml_project.model.train_params import model_params

from ml_project.data_processing.processing import preprocess_data

LOG_CONF_PATH = './configs/logger.conf.yml'
MODEL_PICKLE_PATH = './model/finalized_model.sav'
DATA_PATH = '~/Downloads/heart_cleveland_upload.csv'

logger = logging.getLogger(__name__)


def setup_logging(logging_yaml_config_fpath):
    """setup logging via YAML if it is provided"""
    if logging_yaml_config_fpath:
        with open(logging_yaml_config_fpath) as config_fin:
            logging.config.dictConfig(yaml.safe_load(config_fin))


@click.group()
def main():
    setup_logging(LOG_CONF_PATH)


@main.command(help='train model and save as pickle')
def train():
    logger.debug('start train model...')
    mp = model_params()
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=mp.train_test_split)

    logger.info('preprocess data...')
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    train_y, test_y = train_df.iloc[:, -1].values, test_df.iloc[:, -1].values
    train_x, test_x = train_df.iloc[:, :-1].values, test_df.iloc[:, :-1].values

    logger.info('learn logreg...')
    model = LogisticRegression(penalty=mp.penalty, random_state=mp.random_state)
    model.fit(train_x, train_y)
    y_pred_proba = model.predict_proba(test_x)[:, 1]
    score = roc_auc_score(test_y, y_pred_proba)
    logger.debug(f'ROC AUC score on test = {score}')

    pickle.dump(model, open(MODEL_PICKLE_PATH, 'wb'))
    logger.debug('finish train model')
    return


@main.command()
@click.argument('input_path', type=click.Path())
@click.argument('result_path', type=click.Path())
def predict(input_path, result_path):
    logger.debug('start predict...')
    df = pd.read_csv(input_path)
    df = preprocess_data(df)

    logger.info('load model...')
    loaded_model = pickle.load(open(MODEL_PICKLE_PATH, 'rb'))

    logger.info('predict...')
    predicted_y = loaded_model.predict(df.values)
    np.savetxt(result_path, predicted_y, delimiter='\n')
    logger.debug('finish predict')


if __name__ == "__main__":
    main()




