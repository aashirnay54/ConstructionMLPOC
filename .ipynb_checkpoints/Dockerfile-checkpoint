FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Data/Residential-Building-Data-Set.csv .
COPY train_model.py .
COPY app.py .

RUN python train_model.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
