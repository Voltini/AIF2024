FROM python:3.8
WORKDIR /app
COPY requirements_api.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install gdown &&\
    gdown https://drive.google.com/drive/folders/1B4l6mnXb9YhCTBWeYN6ABSjOiaTK4516?usp=drive_link --folder &&\
    gdown https://drive.google.com/drive/folders/1nhjtQe2eOIEh9aYdMeyruI8DemHxL53t?usp=drive_link --folder &&\
    gdown https://drive.google.com/drive/folders/1QNyaXr_I3fAO3FBPBwcWtK10FYunj6m8?usp=drive_link --folder

EXPOSE 5001
ENV FLASK_ENV=production

COPY api.py helpers.py prep_image.py prep_BERT.py prep_BoW.py /app/
CMD [ "python", "api.py" ]
