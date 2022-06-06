FROM python:3.8-slim-buster

RUN apt update && pip install -U poetry==1.1.8 poetry-core==1.0.4

COPY ml_project/model/finalized_model.sav /finalized_model.sav
COPY ml_project/online_inference/app.py /app.py
COPY pyproject.toml /pyproject.toml

RUN poetry install

WORKDIR .

ENV MODEL_PATH="/finalized_model.sav"

CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]

