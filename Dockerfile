FROM python:3.10-slim

WORKDIR /app

COPY train.py .
CMD ["python", "train.py"]
