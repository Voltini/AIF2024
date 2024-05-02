FROM python:3.12

COPY requirements_webapp.txt requirements.txt
COPY kaggle.json /root/.kaggle/kaggle.json
RUN pip install -r requirements.txt
RUN pip install gdown
RUN kaggle datasets download "ghrzarea/movielens-20m-posters-for-machine-learning" &&\
    unzip movielens-20m-posters-for-machine-learning.zip
WORKDIR /app
RUN gdown https://drive.google.com/drive/folders/1aoipnuUdf03c1lwBwtrWbnBs7q04xRp-?usp=sharing --folder
EXPOSE 80
COPY webapp.py /app/webapp.py

CMD [ "python", "webapp.py" ]