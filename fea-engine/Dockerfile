FROM python:3.10.1-slim-bullseye

WORKDIR /fea-learning

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY ./src ./src

COPY ./data ./data

ENV PYTHONPATH "${PYTHONPATH}:/fea-learning"

EXPOSE 8000

CMD ["python", "./src/engine/engine_api.py"]
#CMD ["/bin/bash"]
#CMD ["uvicorn", "./src/engine/engine_api:app", "--reload"]