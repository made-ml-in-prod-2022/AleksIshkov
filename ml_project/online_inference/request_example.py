import requests


def main():
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    json_data = {
        'data': [
            [69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0, ],
            [68, 0, 1, 140, 239, 0, 0, 151, 1, 1.8, 0, 2, 1, ],
        ],
        'columns': [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
        ],
    }
    response = requests.get('http://0.0.0.0:8001/predict', headers=headers, json=json_data)
    return response


if __name__ == '__main__':
    response = main()
