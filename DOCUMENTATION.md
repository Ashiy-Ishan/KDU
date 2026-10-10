# TensorForge 2.0 MVP — In-Depth System Documentation & Deliverables Guide

**Team**: Team XTurbo  
**System**: RideEat Multilingual Ticket Classification & Routing Engine  
**Version**: 1.2  
**Specification**: OpenAPI 3.0.3 (v2.1.0) / JSON Schema Draft 2020-12  

---

## Table of Contents
1. [System Architecture & Design Decisions](#1-system-architecture--design-decisions)
2. [Machine Learning Pipeline & Methodology](#2-machine-learning-pipeline--methodology)
3. [Domain Precedence & Consistency Rule Engine](#3-domain-precedence--consistency-rule-engine)
4. [Backend API & Asynchronous Job System](#4-backend-api--asynchronous-job-system)
5. [Validation & Test Suite Verification](#5-validation--test-suite-verification)
6. [Public Cloud Deployment Guide](#6-public-cloud-deployment-guide)
7. [15-Minute Video Walkthrough Presentation Script](#7-15-minute-video-walkthrough-presentation-script)
8. [5-Page Technical Report Outline](#8-5-page-technical-report-outline)

---

## 1. System Architecture & Design Decisions

### 1.1 The Challenge
RideEat operates a high-volume multimodal platform (ride-hailing and food delivery) in Sri Lanka, where customer communications exhibit severe linguistic diversity:
* Native Unicode scripts: Sinhala (`si`), Tamil (`ta`).
* Romanized / Transliterated phonetics: Singlish, Tanglish.
* Colloquial and mixed codeswitching (Sinhala/Tamil loanwords blended into English).
* Inherent noise: informal typos, spoken call transcripts (`customer: ... agent: ...`), and emojis.

### 1.2 Architectural Highlights
* **Zero Remote Dependency**: To guarantee deterministic execution, zero latency spikes, and strict offline compliance, the solution runs completely self-contained. Remote LLM calls were explicitly ruled out in accordance with organizer guidelines.
* **Lightweight Standard-Library HTTP Runtime**: Rather than incurring the memory overhead and framework startup latency of heavy async web frameworks, the service leverages Python's multithreaded `ThreadingHTTPServer`. This guarantees:
  * Sub-second container boot times ($< 0.2\text{ s}$).
  * Byte-level control over response headers (`Location`, `Retry-After`, `WWW-Authenticate`, `X-Request-ID`).
  * Strict adherence to error code precedence.
* **Bounded Concurrency & Thread Isolation**: The asynchronous batch job engine processes background evaluations on dedicated threads without impeding health probes or synchronous predictions.

---

## 2. Machine Learning Pipeline & Methodology

### 2.1 Multilingual Feature Extraction
Because customer tickets span native Unicode, English, and Romanized transliterations without language tags provided at inference, word-level tokenizers alone fail on morphological inflections and irregular transliterations.

We utilize a **Feature Union** combining:
1. **Word-level TF-IDF Vectorizer**:
   * `ngram_range=(1, 2)`: Captures domain phrases (`"missing item"`, `"driver late"`, `"refund please"`, `"safety issue"`).
   * `sublinear_tf=True`: Dampens the influence of repeatedly pasted phrases in chat transcripts.
2. **Character Word-Boundary TF-IDF Vectorizer**:
   * `analyzer="char_wb"`, `ngram_range=(3, 5)`: Extracts morphemic roots and phonetic character clusters across Sinhala Unicode, Tamil Unicode, Singlish, and Tanglish.

```
Input: channel + subject + text
   ├──> Word TF-IDF (1-2 n-grams) ──────┐
   └──> Char-wb TF-IDF (3-5 n-grams) ───┴──> Combined Feature Matrix X (~11MB)
```

### 2.2 Model Training Architecture
* **Primary Category Classifier**: Calibrated Linear Support Vector Classifier (`LinearSVC(C=2.0)`) wrapped in `CalibratedClassifierCV`. Linear SVMs maximize the separating margin in sparse high-dimensional text spaces, and probability calibration provides reliable confidence metrics.
* **Secondary Issue Presence Detector**: Calibrated binary classifier identifying whether a ticket contains a secondary intent.
* **Secondary Category Classifier**: Multi-class classifier trained on multi-intent tickets to identify the specific secondary issue.
* **Urgency Classifier**: Calibrated binary classifier with class weighting to reliably detect urgent signals (`acc: 97.0%`, `macro F1: 91.5%`).

### 2.3 Evaluation Results (Official 800-Ticket Holdout Validation Set)
| Metric | Benchmark Score | Target Threshold |
| :--- | :---: | :---: |
| **Primary Category Accuracy** | **65.75%** (was 64.88%) | Baseline |
| **Primary Category Macro F1** | **67.67%** (was 67.42%) | Baseline |
| **Secondary Category Accuracy** | **95.25%** | $> 90\%$ |
| **Urgency Detection Accuracy** | **97.12%** (was 97.00%) | $> 90\%$ |
| **Urgency Macro F1** | **91.69%** (was 91.48%) | $> 85\%$ |
| **Model Artifact Size** | **2.39 MB** (was 11 MB) | $< 100\text{ MB}$ |
| **Single Prediction Latency** | **~0.3 ms** | $< 1,000\text{ ms}$ |
| **100-Ticket Batch Latency** | **~25 ms** | $< 30,000\text{ ms}$ |
| **5,000-Ticket Job Latency** | **~1.8 s** | $< 30\text{ min}$ |

---

## 3. Domain Precedence & Consistency Rule Engine

The system strictly codifies the domain rules specified in `DATA_NOTES.md`:
1. **Safety Precedence**: Safety incidents take primary precedence.
2. **Refund vs. Delay / Missing Items**:
   * If a refund is demanded due to late delivery: Primary $\rightarrow$ `delivery_delay`, Secondary $\rightarrow$ `payment_refund`.
   * If a refund is demanded due to missing food: Primary $\rightarrow$ `order_missing_wrong`, Secondary $\rightarrow$ `payment_refund`.
   * Standalone financial/card charge disputes: Primary $\rightarrow$ `payment_refund`.
3. **Belongings**: Items left in vehicles: Primary $\rightarrow$ `lost_item`.
4. **Spam & Urgency Constraints**:
   * If primary is `spam_irrelevant`, `is_urgent` is strictly `false` and `secondary_category` is strictly `null`.
   * `secondary_category` must never equal `category`.
5. **Team Routing**: 1:1 deterministic mapping:
   ```
   payment_refund        -> Payments & Refunds
   ride_trip_issue       -> Ride Operations
   lost_item             -> Lost & Found
   order_missing_wrong   -> Food Operations
   delivery_delay        -> Delivery Operations
   food_quality          -> Restaurant Quality
   account_promo         -> Account Services
   safety_conduct        -> Trust & Safety
   app_technical         -> Tech Support
   general_inquiry       -> Front-line Support
   spam_irrelevant       -> Auto-close / Spam Filter
   ```

---

## 4. Backend API & Asynchronous Job System

### 4.1 Strict HTTP Error Hierarchy
Every incoming request is processed through a sequential validation filter:
```
1. Authentication (401)
   └── Validates X-API-Key or Authorization Bearer header with constant-time check.
2. Media Type (415)
   └── Validates Content-Type: application/json.
3. Payload Size (413)
   └── Validates body size (1 MB for predict, 5 MB for batch, 25 MB for jobs).
4. JSON Parse (400)
   └── Validates JSON syntax, returns code: "malformed_json".
5. Schema Validation (422)
   └── Validates field types, regex (\S for text), length, uniqueness.
```

### 4.2 Asynchronous Job Lifecycle
* **Job Creation (`POST /batch/jobs`)**:
  * Validates all tickets up front; returns `422` if any ticket is invalid (creating zero jobs).
  * Enforces active concurrency limit (1 running + 3 queued = 4 total active; 5th returns `429 Too Many Requests` with `Retry-After: 10`).
  * Supports idempotency via `Idempotency-Key` header.
  * Returns `202 Accepted` with `Location: /batch/jobs/{job_id}` and `Retry-After: 3`.
* **Job Polling (`GET /batch/jobs/{job_id}`)**:
  * Returns non-decreasing progress (`processed` / `total`) and current state.
  * Includes `Retry-After: 3` header while `queued` or `running`.
* **Results Fetching (`GET /batch/jobs/{job_id}/results`)**:
  * Returns `409 Conflict` if the job is not yet in `succeeded` status.
  * Supports stable pagination (`offset`, `limit`, `next_offset`).
* **Retention & Deletion**:
  * Retains results for 6 hours; returns `410 Gone` after expiration.
  * `DELETE /batch/jobs/{job_id}` cancels a running job or discards results (`204 No Content`). Subsequent queries return `404 Not Found`.

---

## 5. Validation & Test Suite Verification

The automated test runner (`run_tests.sh`) executes end-to-end integration tests:
```bash
./run_tests.sh
```
All tests validate responses against the official JSON Schemas using Draft 2020-12:
* `health_response.schema.json`
* `predict_response.schema.json`
* `batch_response.schema.json`
* `batch_job_status.schema.json`
* `batch_job_results.schema.json`
* `error_response.schema.json`

---

## 6. Public Cloud Deployment Guide

To provide the judges with a public HTTPS endpoint during the evaluation window:

### Option A: Deploy to Google Cloud Run (Recommended)
1. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud config set project <PROJECT_ID>
   ```
2. Build and submit image to Google Artifact Registry:
   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/tensorforge-service:v1.2 .
   ```
3. Deploy container with the assigned API key:
   ```bash
   gcloud run deploy rideeat-support-ai \
       --image gcr.io/<PROJECT_ID>/tensorforge-service:v1.2 \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated \
       --set-env-vars API_KEY=tf2_842bc2c62f969cfe2b8d9be71606bd1a510ede5fc7ffc8e2 \
       --memory 4Gi \
       --cpu 2
   ```
Cloud Run automatically provisions a valid HTTPS URL (e.g. `https://rideeat-support-ai-xyz-uc.a.run.app`).

### Option B: Deploy to AWS EC2 or DigitalOcean Droplet
1. Launch Ubuntu 22.04 LTS (2 vCPU, 4 GB RAM).
2. Install Docker and clone repository:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io
   git clone <REPO_URL> && cd KDU
   ```
3. Run container:
   ```bash
   docker run -d --restart always -p 80:8000 \
       -e API_KEY=tf2_842bc2c62f969cfe2b8d9be71606bd1a510ede5fc7ffc8e2 \
       tensorforge-team-xturbo:v1.2
   ```
4. Configure domain and SSL with Certbot / Caddy for automatic TLS.

---

## 7. 15-Minute Video Walkthrough Presentation Script

### Slide & Timing Breakdown
* **00:00 - 02:00 (Introduction & Problem Statement)**:
  * Present Team XTurbo.
  * Explain RideEat problem statement: multilingual support ticketing across Sinhala, Tamil, English, and Romanized transliterations.
  * Highlight core architectural requirement: 100% offline containerized deployment with zero reliance on LLM wrappers.
* **02:00 - 05:30 (Data Engineering & Multilingual Machine Learning)**:
  * Showcase dataset distribution across languages, channels, and categories.
  * Walk through the Feature Union architecture: why word n-grams combined with character word-boundary n-grams solve Singlish/Tanglish variations without language detection overhead.
  * Explain model selection: Calibrated LinearSVC for multi-class classification and urgency detection.
  * Show validation benchmark metrics: 64.9% category accuracy, 95.5% secondary category accuracy, 97.0% urgency accuracy.
* **05:30 - 08:30 (Domain Precedence & API Architecture)**:
  * Walk through the strict HTTP error hierarchy (`401 -> 415 -> 413 -> 400 -> 422`).
  * Demonstrate the constant-time authentication implementation.
  * Explain the asynchronous batch job engine: worker thread isolation, bounded queue (429 handling), 6-hour retention, and idempotency.
* **08:30 - 12:00 (Live Interactive Demonstration)**:
  * Open web demo dashboard at `http://localhost:8000/`.
  * Test single tickets in Singlish, Sinhala Unicode, and Tamil Unicode; verify instant routing and confidence visualization.
  * Test urgent safety call transcript; verify urgency detection and routing to Trust & Safety.
  * Trigger async bulk job evaluation; demonstrate live progress tracking and paginated results retrieval.
* **12:00 - 14:00 (Offline Container & Test Suite Verification)**:
  * Run `./run_tests.sh` inside container with `--network none`.
  * Highlight 100% pass rate against official JSON Schemas (Draft 2020-12).
* **14:00 - 15:00 (Conclusion & Evaluation URL)**:
  * Summarize strengths: sub-millisecond inference, strict schema compliance, production resilience.
  * Display public HTTPS endpoint for judge evaluation.

---

## 8. 5-Page Technical Report Outline

* **Page 1: Title, Abstract, & Problem Definition**
  * Team XTurbo, RideEat Platform context, objectives, and constraints.
  * Summary of deliverables and key performance figures.
* **Page 2: Multilingual Feature Extraction & ML Modeling**
  * Analysis of morphological complexity in Sinhala and Tamil.
  * FeatureUnion vectorization design (word + character n-grams).
  * Comparative model benchmarks (Logistic Regression, SGD, LinearSVC, Ensembles).
  * Calibration curves and confidence estimation.
* **Page 3: Domain Rules Engine & Consistency Architecture**
  * Business precedence rule formulation (Safety, Delays vs. Refunds, Missing Items).
  * Category-to-Team routing derivation table.
  * Multi-task secondary category detection and urgency classification.
* **Page 4: High-Throughput API & Async Job Engine Design**
  * OpenAPI v2.1.0 compliance architecture.
  * Request validation pipeline and strict error code precedence.
  * Asynchronous job queue, worker pool, idempotency, and retention management.
  * Latency and resource profiling on reference hardware (2 vCPU, 4 GB RAM).
* **Page 5: Evaluation Results, Limitations, & Conclusion**
  * Full classification report on holdout validation dataset.
  * Schema compliance test suite summary.
  * Edge case handling and future improvements.

