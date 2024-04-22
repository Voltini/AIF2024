FROM python:3.8
WORKDIR /app

COPY requirements_api.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 5001
ENV FLASK_ENV=production

ADD https://drive.google.com/uc?export=download&id=1y8M_ed6ZW1ZBPjnye28Q02xUdQD4lIGf /app/rec_imdb.ann 
ADD https://drive.google.com/uc?export=download&id=1zFWmmNmp4aqTPQbaQ-6yC5VsyaGu_Zne /app/model.pth
COPY api.py helpers.py /app/
CMD [ "python", "api.py" ]
