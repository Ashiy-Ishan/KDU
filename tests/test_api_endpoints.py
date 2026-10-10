import os
import sys
import json
import time
import urllib.request
import urllib.error
try:
    import jsonschema
except ImportError:
    jsonschema = None

API_KEY = "tf2_842bc2c62f969cfe2b8d9be71606bd1a510ede5fc7ffc8e2"
BASE_URL = "http://127.0.0.1:8000"
SCHEMAS_DIR = "TensorForge 2.0 - MVP Stage Resources/API specs and schemas"

def load_schema(filename: str) -> dict:
    with open(os.path.join(SCHEMAS_DIR, filename), "r", encoding="utf-8") as f:
        return json.load(f)

def validate_json_schema(instance: dict, schema_filename: str):
    if jsonschema is not None:
        schema = load_schema(schema_filename)
        validator = jsonschema.Draft202012Validator(schema)
        errors = list(validator.iter_errors(instance))
        if errors:
            msg = f"Schema validation failed for {schema_filename}: " + "; ".join(e.message for e in errors)
            raise AssertionError(msg)
    else:
        assert isinstance(instance, dict), "Response must be a dict"
        if "health" in schema_filename:
            assert "status" in instance and "model_version" in instance
        elif "predict_response" in schema_filename:
            for field in ["category", "secondary_category", "team", "is_urgent", "confidence", "model_version"]:
                assert field in instance, f"Missing {field} in predict response"
        elif "error" in schema_filename:
            assert "error" in instance and "code" in instance["error"] and "message" in instance["error"]

def make_request(method: str, path: str, headers: dict = None, body: bytes = None):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url, data=body, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req) as resp:
            resp_body = resp.read()
            resp_headers = dict(resp.headers)
            return resp.status, resp_headers, resp_body
    except urllib.error.HTTPError as e:
        resp_body = e.read()
        resp_headers = dict(e.headers)
        return e.code, resp_headers, resp_body

def run_tests():
    print("=" * 60)
    print("RUNNING TENSORFORGE 2.0 AUTOMATED TEST SUITE")
    print("=" * 60)

    print("\n[1] Testing Public GET /health...")
    status, hdrs, body = make_request("GET", "/health")
    assert status == 200, f"Expected 200, got {status}"
    data = json.loads(body.decode("utf-8"))
    validate_json_schema(data, "health_response.schema.json")
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    print("    PASSED: /health returns 200 and matches HealthResponse schema.")

    print("\n[2] Testing Authentication on POST /predict...")
    status, hdrs, body = make_request("POST", "/predict", headers={"Content-Type": "application/json"}, body=b"{}")
    assert status == 401, f"Expected 401, got {status}"
    assert "Bearer" in hdrs.get("WWW-Authenticate", "")
    err_data = json.loads(body.decode("utf-8"))
    validate_json_schema(err_data, "error_response.schema.json")
    print("    PASSED: Missing API key returns 401 with WWW-Authenticate.")

    status, hdrs, body = make_request("POST", "/predict", headers={"Content-Type": "application/json", "X-API-Key": "wrong_key"}, body=b"{}")
    assert status == 401, f"Expected 401, got {status}"

    ticket = {"channel": "chat", "text": "driver is late refund please", "ticket_id": "TEST-1"}
    body_bytes = json.dumps(ticket).encode("utf-8")
    status, hdrs, body = make_request("POST", "/predict", headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}, body=body_bytes)
    assert status == 200, f"Expected 200, got {status}"
    print("    PASSED: Authorization Bearer header accepted.")

    print("\n[3] Testing HTTP Precedence Order (Auth -> 415 -> 413 -> 400 -> 422)...")
    status, _, body = make_request("POST", "/predict", headers={"X-API-Key": API_KEY, "Content-Type": "text/plain"}, body=b"hello")
    assert status == 415, f"Expected 415, got {status}"
    print("    PASSED: Non-JSON Content-Type returns 415.")

    status, _, body = make_request("POST", "/predict", headers={"X-API-Key": API_KEY, "Content-Type": "application/json"}, body=b"{bad_json")
    assert status == 400, f"Expected 400, got {status}"
    err = json.loads(body.decode("utf-8"))
    assert err["error"]["code"] == "malformed_json"
    print("    PASSED: Malformed JSON syntax returns 400.")

    bad_ticket = {"channel": "chat", "text": "    "}
    status, _, body = make_request("POST", "/predict", headers={"X-API-Key": API_KEY, "Content-Type": "application/json"}, body=json.dumps(bad_ticket).encode("utf-8"))
    assert status == 422, f"Expected 422, got {status}"
    err = json.loads(body.decode("utf-8"))
    validate_json_schema(err, "error_response.schema.json")
    print("    PASSED: Whitespace text returns 422 with details.")

    bad_channel_ticket = {"channel": "carrier_pigeon", "text": "Valid text"}
    status, _, body = make_request("POST", "/predict", headers={"X-API-Key": API_KEY, "Content-Type": "application/json"}, body=json.dumps(bad_channel_ticket).encode("utf-8"))
    assert status == 422, f"Expected 422, got {status}"
    print("    PASSED: Invalid channel returns 422.")

    print("\n[4] Testing POST /predict schema and X-Request-ID echo...")
    req_id = "trace-test-uuid-999"
    test_ticket = {
        "ticket_id": "TF-TRACE-1",
        "channel": "chat",
        "subject": "",
        "text": "කාර් එකේ මගේ බෑග් එක අමතක වුණා, ඩ්‍රයිවර්ට කතා කරන්න පුළුවන්ද?"
    }
    status, hdrs, body = make_request("POST", "/predict", headers={"X-API-Key": API_KEY, "Content-Type": "application/json", "X-Request-ID": req_id}, body=json.dumps(test_ticket).encode("utf-8"))
    assert status == 200
    assert hdrs.get("X-Request-ID") == req_id
    pred_res = json.loads(body.decode("utf-8"))
    validate_json_schema(pred_res, "predict_response.schema.json")
    assert pred_res["ticket_id"] == "TF-TRACE-1"
    assert pred_res["team"] == "Lost & Found" or pred_res["category"] == "lost_item"
    print("    PASSED: POST /predict returns valid PredictResponse with echoed X-Request-ID.")

    print("\n[5] Testing POST /predict/batch (Synchronous 1-100 tickets)...")
    batch_payload = {
        "tickets": [
            {"ticket_id": "T-1", "channel": "chat", "text": "late order refund"},
            {"ticket_id": "T-2", "channel": "email", "subject": "Item missing", "text": "Burger missing from delivery"},
            {"ticket_id": "T-3", "channel": "call_transcript", "text": "driver refused to stop car I am scared"}
        ]
    }
    status, hdrs, body = make_request("POST", "/predict/batch", headers={"X-API-Key": API_KEY, "Content-Type": "application/json"}, body=json.dumps(batch_payload).encode("utf-8"))
    assert status == 200, f"Expected 200, got {status}"
    batch_res = json.loads(body.decode("utf-8"))
    validate_json_schema(batch_res, "batch_response.schema.json")
    assert len(batch_res["predictions"]) == 3
    assert batch_res["predictions"][0]["ticket_id"] == "T-1"
    assert batch_res["predictions"][1]["ticket_id"] == "T-2"
    assert batch_res["predictions"][2]["ticket_id"] == "T-3"
    print("    PASSED: Synchronous batch prediction preserves order and passes BatchResponse schema.")

    bad_batch = {
        "tickets": [
            {"ticket_id": "DUP-1", "channel": "chat", "text": "first item"},
            {"ticket_id": "DUP-1", "channel": "chat", "text": "duplicate item"}
        ]
    }
    status, _, body = make_request("POST", "/predict/batch", headers={"X-API-Key": API_KEY, "Content-Type": "application/json"}, body=json.dumps(bad_batch).encode("utf-8"))
    assert status == 422, f"Expected 422, got {status}"
    err_b = json.loads(body.decode("utf-8"))
    assert any(d.get("issue") == "duplicate of an earlier item" for d in err_b["error"].get("details", []))
    print("    PASSED: Duplicate ticket_id in batch fails atomically with 422.")

    print("\n[6] Testing Async Bulk Evaluation Engine (/batch/jobs)...")
    job_tickets = [
        {"ticket_id": f"JOB-TICK-{i}", "channel": "chat", "text": f"Ticket number {i} text for processing"}
        for i in range(1, 21)
    ]
    status, hdrs, body = make_request("POST", "/batch/jobs", headers={"X-API-Key": API_KEY, "Content-Type": "application/json"}, body=json.dumps({"tickets": job_tickets}).encode("utf-8"))
    assert status == 202, f"Expected 202, got {status}"
    assert "Location" in hdrs
    assert "Retry-After" in hdrs
    job_status = json.loads(body.decode("utf-8"))
    validate_json_schema(job_status, "batch_job_status.schema.json")
    job_id = job_status["job_id"]
    print(f"    PASSED: Job created with ID: {job_id}, status: {job_status['status']}.")

    print("    Polling job status...")
    max_wait = 10
    start = time.time()
    succeeded = False
    while time.time() - start < max_wait:
        status, poll_hdrs, body = make_request("GET", f"/batch/jobs/{job_id}", headers={"X-API-Key": API_KEY})
        assert status == 200
        cur_status = json.loads(body.decode("utf-8"))
        validate_json_schema(cur_status, "batch_job_status.schema.json")
        if cur_status["status"] == "succeeded":
            succeeded = True
            assert cur_status["processed"] == 20
            break
        time.sleep(0.3)
    assert succeeded, "Job did not complete within timeout"
    print("    PASSED: Job reached succeeded state with 20/20 tickets processed.")

    print("    Fetching job results...")
    status, _, body = make_request("GET", f"/batch/jobs/{job_id}/results?offset=0&limit=10", headers={"X-API-Key": API_KEY})
    assert status == 200
    res_data = json.loads(body.decode("utf-8"))
    validate_json_schema(res_data, "batch_job_results.schema.json")
    assert len(res_data["predictions"]) == 10
    assert res_data["next_offset"] == 10
    print("    PASSED: Results fetch and pagination conforms to BatchJobResults schema.")

    print("    Cancelling/Discarding job via DELETE...")
    status, _, _ = make_request("DELETE", f"/batch/jobs/{job_id}", headers={"X-API-Key": API_KEY})
    assert status == 204
    status, _, _ = make_request("GET", f"/batch/jobs/{job_id}", headers={"X-API-Key": API_KEY})
    assert status == 404
    print("    PASSED: DELETE returns 204 and subsequent queries return 404.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
