# TensorForge 2.0 MVP — RideEat Ticket Classification & Routing Engine

**Team Name**: Team XTurbo  
**Model Version**: `v1.2`  
**API Specification**: OpenAPI 3.0.3 (v2.1.0) / JSON Schema Draft 2020-12  

---

## 1. Quick Start (Official Docker Run)

In accordance with the competition guidelines, the entire solution runs inside a single offline Docker container with **no internet access required at runtime**:

```bash
docker run -p 8000:8000 -e API_KEY=<API_KEY> tensorforge-team-xturbo:v1.2
```

Once running:
- **Web UI & Live Triage Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **Public Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 2. Building and Running with Docker

### Step 1: Build the Docker Image
```bash
docker build -t tensorforge-team-xturbo:v1.2 .
```

### Step 2: Run the Container
```bash
docker run -d \
  --name tensorforge-service \
  -p 8000:8000 \
  -e API_KEY=<API_KEY> \
  tensorforge-team-xturbo:v1.2
```

### Or Use the Convenience Script:
```bash
./run_container.sh
```

To stop the running container:
```bash
docker stop tensorforge-service && docker rm tensorforge-service
```

---

## 3. Running Automated Test Suite

To verify the service against the official JSON Schemas (Draft 2020-12) and HTTP error precedence rules inside an isolated offline environment (`--network none`):

```bash
./run_tests.sh
```

This runs `tests/test_api_endpoints.py` which validates:
1. Public `GET /health` schema response.
2. Authentication checks (`401 Unauthorized` with `WWW-Authenticate: Bearer`).
3. HTTP Status Code Precedence (`401 -> 415 -> 413 -> 400 -> 422`).
4. Single-ticket inference and `X-Request-ID` tracing header echo.
5. Synchronous batch prediction (1–100 items) and duplicate ticket rejection.
6. Asynchronous bulk evaluation engine (`202 Accepted`, polling, pagination, and `204 No Content` deletion).

---

## 4. Running Locally (Without Docker)

### Prerequisites
- Python 3.11+
- Dependencies: `scikit-learn`, `numpy`, `scipy`, `pandas`, `joblib`, `jsonschema`

### Step 1: Set up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install scikit-learn numpy scipy pandas joblib jsonschema
```

### Step 2: Start the Server
```bash
export API_KEY="<API_KEY>"
export PORT=8000
python3 src/api/server.py
```

---

## 5. Training the Machine Learning Model

To retrain the multilingual model pipeline from scratch and export the updated artifact:

```bash
python3 src/ml/train.py
```

This script:
1. Loads `train.csv` (4,000 tickets) and `validation.csv` (800 tickets).
2. Fits the multilingual `FeatureUnion` (Word n-grams 1–2 + Character word-boundary n-grams 3–5).
3. Fits the Primary, Secondary, and Urgency Linear SVM classifiers with temperature-scaled Softmax calibration.
4. Serializes the pipeline to [`src/models/model_v1.2.joblib`](src/models/model_v1.2.joblib).
5. Outputs evaluation metrics to [`src/models/validation_metrics.json`](src/models/validation_metrics.json).

---

## 6. API Usage & Curl Examples

### 1. Health Check (Public)
```bash
curl -X GET http://localhost:8000/health
```
**Response**:
```json
{
  "status": "ok",
  "model_version": "v1.2",
  "model_loaded": true
}
```

---

### 2. Single Ticket Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "ticket_id": "TEST-001",
    "channel": "chat",
    "subject": "",
    "text": "bro mage order eka hour ekakata wada late, refund ekak denna puluwanda?"
  }'
```
*(Also supports `Authorization: Bearer <API_KEY>`)*

**Response**:
```json
{
  "category": "delivery_delay",
  "secondary_category": "payment_refund",
  "team": "Delivery Operations",
  "is_urgent": false,
  "confidence": 0.8842,
  "model_version": "v1.2",
  "needs_human_review": false,
  "ticket_id": "TEST-001"
}
```

---

### 3. Synchronous Batch Prediction (`POST /predict/batch`)
Evaluates 1 to 100 tickets synchronously:
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "tickets": [
      {
        "ticket_id": "T-1",
        "channel": "chat",
        "text": "කාර් එකේ මගේ බෑග් එක අමතක වුණා"
      },
      {
        "ticket_id": "T-2",
        "channel": "call_transcript",
        "text": "customer: driver driving recklessly refusing to stop please help"
      }
    ]
  }'
```

---

### 4. Asynchronous Bulk Evaluation Job (`POST /batch/jobs`)
For large evaluation runs (up to 5,000 tickets):

**Submit Job**:
```bash
curl -i -X POST http://localhost:8000/batch/jobs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <API_KEY>" \
  -d '{
    "tickets": [ ... up to 5000 tickets ... ]
  }'
```
Returns HTTP `202 Accepted` with `Location: /batch/jobs/{job_id}` and `Retry-After: 3`.

**Poll Status**:
```bash
curl -X GET http://localhost:8000/batch/jobs/<job_id> \
  -H "X-API-Key: <API_KEY>"
```

**Fetch Results (with pagination)**:
```bash
curl -X GET "http://localhost:8000/batch/jobs/<job_id>/results?offset=0&limit=50" \
  -H "X-API-Key: <API_KEY>"
```

**Cancel / Discard Job**:
```bash
curl -i -X DELETE http://localhost:8000/batch/jobs/<job_id> \
  -H "X-API-Key: <API_KEY>"
```
Returns HTTP `204 No Content`.

---

## 7. Project Structure

```
├── Dockerfile                  # Container definition (offline runtime)
├── REPORT.md                   # 5-Page Technical Submission Report
├── README.md                   # System running documentation
├── run_container.sh            # One-click Docker container runner
├── run_tests.sh                # Isolated test suite runner (--network none)
├── src/
│   ├── api/
│   │   ├── auth.py             # Constant-time API key verification
│   │   ├── job_manager.py      # Async background job engine (5k tickets)
│   │   ├── schemas.py          # Draft 2020-12 schema validation rules
│   │   └── server.py           # Threaded HTTP server & routing
│   ├── ml/
│   │   ├── predictor.py        # Sub-0.5ms inference & domain rules engine
│   │   └── train.py            # FeatureUnion & LinearSVM training pipeline
│   ├── models/
│   │   ├── model_v1.2.joblib   # Serialized model artifact (2.39 MB)
│   │   └── validation_metrics.json
│   └── web/
│       └── index.html          # Minimalist offline dashboard
└── tests/
    └── test_api_endpoints.py   # E2E integration test suite
```
