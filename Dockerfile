FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN python train.py

EXPOSE 5000

CMD ["python", "app.py"]
