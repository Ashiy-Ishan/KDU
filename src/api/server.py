import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import traceback

from src.api.auth import verify_request_auth
from src.api.schemas import validate_single_ticket, validate_batch_tickets
from src.api.job_manager import JobManager
from src.ml.predictor import TicketPredictor

LIMIT_PREDICT = 1 * 1024 * 1024
LIMIT_BATCH_PREDICT = 5 * 1024 * 1024
LIMIT_BATCH_JOBS = 25 * 1024 * 1024

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class TensorForgeRequestHandler(BaseHTTPRequestHandler):
    server_version = "TensorForge/2.1.0"

    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.command} {self.path} - {format % args}\n")

    def send_json(self, status_code: int, data: dict, headers: dict = None):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        
        client_req_id = self.headers.get("X-Request-ID")
        if client_req_id:
            self.send_header("X-Request-ID", client_req_id.strip())

        if headers:
            for k, v in headers.items():
                self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def send_error_response(self, status_code: int, code: str, message: str, details: list = None, extra_headers: dict = None):
        payload = {
            "error": {
                "code": code,
                "message": message
            }
        }
        if details:
            payload["error"]["details"] = details
        self.send_json(status_code, payload, headers=extra_headers)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/health":
            self.handle_health()
            return

        if path in ("/", "/dashboard"):
            self.handle_dashboard()
            return

        is_auth, auth_err = verify_request_auth(dict(self.headers))
        if not is_auth:
            self.send_error_response(401, "unauthorized", auth_err, extra_headers={"WWW-Authenticate": "Bearer"})
            return

        if path.startswith("/batch/jobs/") and path.endswith("/results"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[0] == "batch" and parts[1] == "jobs" and parts[3] == "results":
                job_id = parts[2]
                self.handle_get_job_results(job_id, query)
                return

        if path.startswith("/batch/jobs/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "batch" and parts[1] == "jobs":
                job_id = parts[2]
                self.handle_get_job_status(job_id)
                return

        self.send_error_response(404, "not_found", f"Route GET {path} not found.")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/predict", "/predict/batch", "/batch/jobs"):
            is_auth, auth_err = verify_request_auth(dict(self.headers))
            if not is_auth:
                self.send_error_response(401, "unauthorized", auth_err, extra_headers={"WWW-Authenticate": "Bearer"})
                return

            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/json"):
                self.send_error_response(415, "unsupported_media_type", "Content-Type must be application/json.")
                return

            try:
                content_length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                self.send_error_response(400, "malformed_json", "Invalid Content-Length header.")
                return

            max_limit = LIMIT_PREDICT
            if path == "/predict/batch":
                max_limit = LIMIT_BATCH_PREDICT
            elif path == "/batch/jobs":
                max_limit = LIMIT_BATCH_JOBS

            if content_length > max_limit:
                self.send_error_response(413, "payload_too_large", f"Request body exceeds maximum allowed size of {max_limit} bytes.")
                return

            raw_body = self.rfile.read(content_length)

            try:
                body_data = json.loads(raw_body.decode("utf-8"))
            except Exception:
                self.send_error_response(400, "malformed_json", "Request body is not valid JSON.")
                return

            if path == "/predict":
                self.handle_predict(body_data)
            elif path == "/predict/batch":
                self.handle_predict_batch(body_data)
            elif path == "/batch/jobs":
                self.handle_submit_batch_job(body_data)
            return

        self.send_error_response(404, "not_found", f"Route POST {path} not found.")

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        is_auth, auth_err = verify_request_auth(dict(self.headers))
        if not is_auth:
            self.send_error_response(401, "unauthorized", auth_err, extra_headers={"WWW-Authenticate": "Bearer"})
            return

        if path.startswith("/batch/jobs/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "batch" and parts[1] == "jobs":
                job_id = parts[2]
                self.handle_delete_job(job_id)
                return

        self.send_error_response(404, "not_found", f"Route DELETE {path} not found.")

    def handle_health(self):
        pred = self.server.predictor
        if pred and pred.is_loaded:
            self.send_json(200, {
                "status": "ok",
                "model_version": pred.model_version,
                "model_loaded": True
            })
        else:
            self.send_json(503, {
                "status": "loading",
                "model_version": None,
                "model_loaded": False
            })

    def handle_predict(self, data: dict):
        issues = validate_single_ticket(data, require_ticket_id=False)
        if issues:
            self.send_error_response(422, "validation_error", "Request failed validation.", details=issues)
            return

        channel = data.get("channel")
        subject = data.get("subject", "")
        text = data.get("text")
        ticket_id = data.get("ticket_id")

        pred = self.server.predictor
        res = pred.predict_one(channel, subject, text, ticket_id=ticket_id)
        self.send_json(200, res)

    def handle_predict_batch(self, data: dict):
        start_time = time.perf_counter()
        if not isinstance(data, dict) or "tickets" not in data:
            self.send_error_response(422, "validation_error", "Request failed validation.", details=[{"field": "tickets", "issue": "is required"}])
            return

        tickets = data.get("tickets")
        is_valid, issues = validate_batch_tickets(tickets, max_items=100)
        if not is_valid:
            self.send_error_response(422, "validation_error", "Request failed validation.", details=issues)
            return

        pred = self.server.predictor
        results = pred.predict_batch(tickets)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        response_payload = {
            "predictions": results,
            "meta": {
                "count": len(results),
                "model_version": pred.model_version,
                "processing_time_ms": elapsed_ms
            }
        }
        self.send_json(200, response_payload)

    def handle_submit_batch_job(self, data: dict):
        if not isinstance(data, dict) or "tickets" not in data:
            self.send_error_response(422, "validation_error", "Request failed validation.", details=[{"field": "tickets", "issue": "is required"}])
            return

        tickets = data.get("tickets")
        is_valid, issues = validate_batch_tickets(tickets, max_items=5000)
        if not is_valid:
            self.send_error_response(422, "validation_error", "Request failed validation.", details=issues)
            return

        idempotency_key = self.headers.get("Idempotency-Key")
        if idempotency_key:
            idempotency_key = idempotency_key.strip()

        job_mgr = self.server.job_manager
        status_code, resp_body, job_id = job_mgr.submit_job(tickets, idempotency_key=idempotency_key)

        if status_code == 429:
            self.send_json(429, resp_body, headers={"Retry-After": 10})
        else:
            headers = {
                "Location": f"/batch/jobs/{job_id}",
                "Retry-After": 3
            }
            self.send_json(202, resp_body, headers=headers)

    def handle_get_job_status(self, job_id: str):
        job_mgr = self.server.job_manager
        status_code, resp_body = job_mgr.get_job_status(job_id)

        headers = {}
        if status_code == 200 and resp_body.get("status") in ("queued", "running"):
            headers["Retry-After"] = 3

        self.send_json(status_code, resp_body, headers=headers)

    def handle_get_job_results(self, job_id: str, query: dict):
        offset = 0
        limit = None

        if "offset" in query:
            try:
                offset = int(query["offset"][0])
                if offset < 0:
                    self.send_error_response(422, "validation_error", "Query param 'offset' must be >= 0.")
                    return
            except ValueError:
                self.send_error_response(422, "validation_error", "Query param 'offset' must be integer.")
                return

        if "limit" in query:
            try:
                limit = int(query["limit"][0])
                if limit < 1 or limit > 5000:
                    self.send_error_response(422, "validation_error", "Query param 'limit' must be between 1 and 5000.")
                    return
            except ValueError:
                self.send_error_response(422, "validation_error", "Query param 'limit' must be integer.")
                return

        job_mgr = self.server.job_manager
        status_code, resp_body = job_mgr.get_job_results(job_id, offset=offset, limit=limit)
        self.send_json(status_code, resp_body)

    def handle_delete_job(self, job_id: str):
        job_mgr = self.server.job_manager
        status_code, resp_body = job_mgr.delete_job(job_id)
        if status_code == 204:
            self.send_response(204)
            client_req_id = self.headers.get("X-Request-ID")
            if client_req_id:
                self.send_header("X-Request-ID", client_req_id.strip())
            self.end_headers()
        else:
            self.send_json(status_code, resp_body)

    def handle_dashboard(self):
        dashboard_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "index.html")
        if os.path.exists(dashboard_path):
            with open(dashboard_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_json(200, {
                "service": "TensorForge 2.0 MVP Ticket Classification API",
                "status": "online",
                "docs": "/health"
            })

def run_server(port: int = 8000, host: str = "0.0.0.0"):
    print(f"Initializing TensorForge 2.0 Model Predictor...")
    predictor = TicketPredictor()
    predictor.load()

    print(f"Initializing Async Job Manager...")
    job_manager = JobManager(predictor)

    server = ThreadedHTTPServer((host, port), TensorForgeRequestHandler)
    server.predictor = predictor
    server.job_manager = job_manager

    print(f"TensorForge 2.0 API Server running at http://{host}:{port}/")
    print(f"   • Public Health Check: http://{host}:{port}/health")
    print(f"   • Live Interactive Demo: http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        server.server_close()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    run_server(port=port)
