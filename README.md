# TensorForge 2.0 (Phase 2 MVP) — RideEat Support AI

[![OpenAPI v2.1.0](https://img.shields.io/badge/OpenAPI-v2.1.0-blue.svg)](file:///home/dark-devil/Desktop/git_clones/KDU/TensorForge%202.0%20-%20MVP%20Stage%20Resources/API%20specs%20and%20schemas/tensorforge-phase2-openapi-v2.yaml)
[![Status](https://img.shields.io/badge/Build-Passing-emerald.svg)]()
[![Model Version](https://img.shields.io/badge/Model-v1.2-indigo.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Offline--Ready-cyan.svg)]()

> **Team Name**: Team XTurbo  
> **Challenge**: TensorForge 2.0 — Phase 2 MVP Stage  
> **Problem Statement**: Multilingual Customer Support Ticket Classification & Automated Routing System for **RideEat**

---

## 1. Project Overview

RideEat Forge AI is an offline-capable, high-throughput customer support ticket classification and routing system. It processes noisy, multilingual support tickets across English, Sinhala Unicode, Tamil Unicode, Singlish (transliterated Sinhala), Tanglish (transliterated Tamil), and mixed scripts.

The system performs:
1. **Primary Issue Classification (`category`)**: Predicts one of 11 distinct operational categories.
2. **Secondary Issue Detection (`secondary_category`)**: Detects secondary issues for multi-intent tickets; strictly `null` for spam.
3. **Urgency Flagging (`is_urgent`)**: Identifies immediate threats, acute medical events, fraud, and active driver safety incidents.
4. **Deterministic Team Derivation (`team`)**: Routes tickets to the designated team via the mandatory routing table.
5. **Confidence & Triage Scoring (`confidence`, `needs_human_review`)**: Calibrated confidence scores with automated fallback flags for human review.

---

## 2. Quickstart: Run in Docker (Offline)

As specified in the competition guidelines, the service boots with a single command and **requires zero internet access at runtime**:

```bash
docker run -p 8000:8000 -e API_KEY=tf2_842bc2c62f969cfe2b8d9be71606bd1a510ede5fc7ffc8e2 tensorforge-team-xturbo:v1.2
```

Or using the bundled shell script:
```bash
./run_container.sh
```

### Accessing the Service
* **Public Health Check**: [http://localhost:8000/health](http://localhost:8000/health) (Ready in $< 2\text{ s}$)
* **Interactive Live Demo Dashboard**: [http://localhost:8000/](http://localhost:8000/)
* **Prediction Endpoint**: `POST http://localhost:8000/predict`
* **Synchronous Batch Endpoint**: `POST http://localhost:8000/predict/batch`
* **Async Bulk Job Endpoint**: `POST http://localhost:8000/batch/jobs`

---

## 3. Architecture & Tech Stack

```
                          +------------------------------------------------+
                          |             JUDGE EVALUATION HARNESS           |
                          +------------------------------------------------+
                                       |                     |
                   Public Health Check |                     | API Key Protected
                                       v                     v
                               +---------------+    +--------------------+
                               |  GET /health  |    |  /predict          |
                               +---------------+    |  /predict/batch    |
                                                    |  /batch/jobs       |
                                                    +--------------------+
                                                              |
                                                              v
                                              +--------------------------------+
                                              | Strict HTTP Precedence Filter  |
                                              |  1. Auth Check (401)           |
                                              |  2. Media Type Check (415)     |
                                              |  3. Size Limits (413)          |
                                              |  4. JSON Parse Check (400)     |
                                              |  5. Schema Validation (422)    |
                                              +--------------------------------+
                                                              |
                                                              v
                     +----------------------------------------+----------------------------------------+
                     |                                                                                 |
                     v                                                                                 v
         +-----------------------+                                                         +-----------------------+
         |  Synchronous Engine   |                                                         |    Async Job Engine   |
         |  - /predict (< 1s)    |                                                         |  - Non-blocking queue |
         |  - /predict/batch     |                                                         |  - Location & Retry   |
         +-----------------------+                                                         |  - Concurrency (429)  |
                     |                                                                     |  - 6-Hour Retention   |
                     v                                                                     +-----------------------+
         +-----------------------------------------------------------------------------+               |
         |                         ML Predictor & Business Engine                      |<--------------+
         |  • Multilingual FeatureUnion (Word n-grams 1-2 + Char-wb n-grams 3-5)       |
         |  • Calibrated LinearSVC Classifier (11 Categories, Acc: 64.9%, F1: 67.4%)   |
         |  • Secondary Issue Detector & Classifier (Overall Acc: 95.5%)               |
         |  • Calibrated Urgency Detector (Acc: 97.0%, Macro F1: 91.5%)                |
         |  • Domain Consistency Rules & Category-to-Team Derivation                  |
         +-----------------------------------------------------------------------------+
```

### Key Technical Decisions
* **No LLM Wrappers**: Completely offline classical and calibrated machine learning models with reproducible feature extraction.
* **Standard Library HTTP Server**: Built with `http.server.ThreadingHTTPServer` to ensure sub-second startup ($< 0.2\text{ s}$), zero package compilation at runtime, and strict control over HTTP headers and error response precedence.
* **Bounded Concurrency & Background Worker**: Handles up to 5,000 tickets per job asynchronously while keeping `/health` and live `/predict` traffic sub-second responsive.

---

## 4. API Endpoints Reference

| Method | Endpoint | Auth Required | Description |
| :--- | :--- | :---: | :--- |
| `GET` | `/health` | No | Public health check. Returns `200` with model version when ready, `503` when loading. |
| `GET` | `/` or `/dashboard` | No | Embedded modern HTML5/JS demo dashboard for live evaluations. |
| `POST` | `/predict` | **Yes** | Single ticket classification. Max payload $1\text{ MB}$. |
| `POST` | `/predict/batch` | **Yes** | Synchronous batch classification ($1\text{ to }100$ tickets). Max payload $5\text{ MB}$. Atomic validation. |
| `POST` | `/batch/jobs` | **Yes** | Submit asynchronous bulk evaluation job ($1\text{ to }5,000$ tickets). Returns `202 Accepted` with `Location` and `Retry-After`. |
| `GET` | `/batch/jobs/{job_id}` | **Yes** | Poll async job status and progress (`queued`, `running`, `succeeded`, `failed`, `cancelled`). |
| `GET` | `/batch/jobs/{job_id}/results` | **Yes** | Retrieve paginated predictions once succeeded (supports `offset` and `limit`). |
| `DELETE`| `/batch/jobs/{job_id}` | **Yes** | Cancel running job or discard completed results (`204 No Content`). Subsequent queries return `404`. |

---

## 5. Automated Testing

Run the full automated test suite verifying all status codes, error precedence, headers, and official JSON schemas:

```bash
./run_tests.sh
```

Tests cover:
* Public `/health` schema compliance
* Authentication rejection (missing key $\rightarrow$ `401` with `WWW-Authenticate: Bearer`)
* Media type validation (`Content-Type: text/plain` $\rightarrow$ `415`)
* Payload size limits ($> 1\text{ MB}$ $\rightarrow$ `413`)
* Malformed JSON syntax $\rightarrow$ `400`
* Empty / whitespace text $\rightarrow$ `422` with structured details
* Duplicate `ticket_id` detection in batches $\rightarrow$ `422`
* Asynchronous job lifecycle: submission $\rightarrow$ `202`, progress tracking $\rightarrow$ `succeeded`, results pagination $\rightarrow$ `200`, cancellation $\rightarrow$ `204`, subsequent lookup $\rightarrow$ `404`

---

## 6. Model Training & Evaluation

To retrain the model locally and regenerate artifacts:

```bash
python3 src/ml/train.py
```

### Validation Benchmark Scores (Official 800-Ticket Validation Set)
* **Primary Category Accuracy**: `65.75%` (Improved from 64.88%)
* **Primary Category Macro F1**: `67.67%` (Improved from 67.42%)
* **Secondary Category Accuracy**: `95.25%`
* **Urgency Classification Accuracy**: `97.12%` (Improved from 97.00%)
* **Urgency Macro F1**: `91.69%` (Improved from 91.48%)
* **Model Artifact Size**: `2.39 MB` (Reduced by 78% from 11 MB)
* **Single Ticket Inference Latency**: `~0.3 ms` (Benchmark requirement: $< 1,000\text{ ms}$)
* **100-Ticket Batch Latency**: `~25 ms` (Benchmark requirement: $< 30,000\text{ ms}$)
* **5,000-Ticket Async Job Latency**: `~1.8 s` (Benchmark requirement: $< 30\text{ min}$)

---

## 7. Submission Checklist Verification

- [x] **Service validated locally against JSON Schemas** (Draft 2020-12 compliant).
- [x] **API Key Security**: Prediction and job endpoints enforce `API_KEY`; `/health` and `/` remain public.
- [x] **Docker Container**: Starts with `docker run -p 8000:8000 -e API_KEY=<key> <image>` with zero internet dependency.
- [x] **Batch Inference Endpoint**: Fully compliant with OpenAPI v2.1.0 (status codes, headers, and schemas).
- [x] **Live Demo Dashboard**: Accessible directly at the root URL.
- [x] **Machine Learning Rationale**: Transparent training script, reproducible artifacts, and no third-party LLM wrapper.

