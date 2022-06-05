from unittest.mock import patch
from unittest.mock import ANY
import pandas as pd
import pytest
from click.testing import CliRunner
import numpy as np

from ml_project.model.train_params import model_params
from ml_project.data_processing.processing import preprocess_data
from ml_project.model.fit_predict import main

@patch('yaml.safe_load')
def test_load_params(yaml_load):
    yaml_load.return_value = {
        'prod': {
            'train_test_split': 0.3,
            'penalty': 'l2',
            'random_state': 42,
        },
    }
    x = model_params.load_params_from_file('prod')
    assert x == {'penalty': 'l2', 'random_state': 42, 'train_test_split': 0.3}


@pytest.fixture()
def data_sample():
    data = {
        'age': {0: 69, 1: 68},
        'sex': {0: 1, 1: 0},
        'cp': {0: 0, 1: 0},
        'trestbps': {0: 160, 1: 140},
        'chol': {0: 234, 1: 239},
        'fbs': {0: 1, 1: 0},
        'restecg': {0: 2, 1: 0},
        'thalach': {0: 131, 1: 151},
        'exang': {0: 0, 1: 0},
        'oldpeak': {0: 0.1, 1: 1.8},
        'slope': {0: 1, 1: 0},
        'ca': {0: 1, 1: 2},
        'thal': {0: 0, 1: 0},
        'condition': {0: 0, 1: 0}
    }
    return pd.DataFrame(data)


def test_preprocess_data(data_sample):
    df = preprocess_data(data_sample)
    assert len(df.columns) == 29


def test_predict(data_sample):
    runner = CliRunner()
    with patch('pandas.read_csv') as pd_read:
        pd_read.return_value = data_sample.iloc[:,:-1]
        with patch('numpy.savetxt') as np_write:
             _ = runner.invoke(
                 main,
                 [
                     'predict',
                     'input_fpath',
                     'output_fpath',
                  ]
             )
    np_write.assert_called_once_with('output_fpath', ANY, delimiter='\n')
