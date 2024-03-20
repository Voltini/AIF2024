FROM python:3.12

COPY requirements_webapp.txt requirements.txt
RUN pip install -r requirements.txt
COPY kaggle.json /root/.kaggle/kaggle.json
RUN kaggle datasets download "ghrzarea/movielens-20m-posters-for-machine-learning" &&\
    unzip movielens-20m-posters-for-machine-learning.zip
EXPOSE 80
WORKDIR /app
COPY webapp.py /app/webapp.py

CMD [ "python", "webapp.py" ]