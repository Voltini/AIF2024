FROM python:3.8
WORKDIR /app

COPY requirements_api.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 5000
ENV FLASK_ENV=production

ADD https://drive.google.com/uc?export=download&id=1eSRxYIIgiEziQNoGTv0Lt2K1x-2fW5Mz /app/annoy_index.ann 
ADD https://drive.google.com/uc?export=download&id=19Iz0ksLO5LBHxf4QG3w1ty8jqQUA9Ji7 /app/model.pth
COPY api.py helpers.py /app/
CMD [ "python", "api.py" ]
