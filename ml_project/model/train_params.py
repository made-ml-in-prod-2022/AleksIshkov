import yaml

PATH = './configs/model_params.yml'


class model_params:
    train_test_split: float
    penalty: str
    random_state: int

    def __init__(self, env='prod'):
        params_dict = model_params.load_params_from_file(env)
        self.train_test_split = params_dict['train_test_split']
        self.penalty = params_dict['penalty']
        self.random_state = params_dict['random_state']

    @staticmethod
    def load_params_from_file(env):
        with open(PATH) as mp_fio:
            raw_yaml = yaml.safe_load(mp_fio)
        return raw_yaml[env]

