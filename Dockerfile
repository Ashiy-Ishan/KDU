# TensorForge 2.0 MVP Stage - Production Container
# Team: Team XTurbo
# Offline-capable, zero runtime dependencies, self-contained model artifacts

FROM ml:1

WORKDIR /app

# Copy model artifacts, application source, dashboard, and schemas
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY ["TensorForge 2.0 - MVP Stage Resources/API specs and schemas", "/app/TensorForge 2.0 - MVP Stage Resources/API specs and schemas"]

# Environment configuration
ENV PORT=8000
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose standard evaluation port
EXPOSE 8000

# Container entrypoint starts server directly with no internet access required
ENTRYPOINT ["python3.11", "/app/src/api/server.py"]
