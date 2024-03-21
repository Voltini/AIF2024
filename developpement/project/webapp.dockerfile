FROM python:3.12

COPY requirements_webapp.txt requirements.txt
RUN pip install -r requirements.txt
COPY kaggle.json /root/.kaggle/kaggle.json
RUN kaggle datasets download "ghrzarea/movielens-20m-posters-for-machine-learning" &&\
    unzip movielens-20m-posters-for-machine-learning.zip
EXPOSE 80
WORKDIR /app
COPY webapp.py /app/webapp.py
ADD https://drive.google.com/uc?export=download&id=1hWsOTci3ZovpA4QSPUOO_cZ53QqVzXzJ /app/feature-path.pickle 

CMD [ "python", "webapp.py" ]