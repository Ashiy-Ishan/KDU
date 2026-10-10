FROM ml:1

WORKDIR /app

COPY src/ /app/src/
COPY tests/ /app/tests/
COPY ["TensorForge 2.0 - MVP Stage Resources/API specs and schemas", "/app/TensorForge 2.0 - MVP Stage Resources/API specs and schemas"]

ENV PORT=8000
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["python3.11", "/app/src/api/server.py"]
