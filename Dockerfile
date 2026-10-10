FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY tests/ /app/tests/
COPY ["TensorForge 2.0 - MVP Stage Resources/API specs and schemas", "/app/TensorForge 2.0 - MVP Stage Resources/API specs and schemas"]

ENV PORT=8000
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["python3", "/app/src/api/server.py"]
