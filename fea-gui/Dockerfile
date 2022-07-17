FROM python:3.10.1-slim-bullseye

WORKDIR /fea-gui

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY ./src ./src

ENV PYTHONPATH "${PYTHONPATH}:/fea-gui"
# ENV API_IP=${API_IP}
EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["./src/gui_app.py"]
#, "10.100.102.98"]
#CMD ["./src/gui_app.py", "10.100.102.98"]

#CMD ["streamlit", "run", "./src/gui_app.py"]
#, "localhost"]