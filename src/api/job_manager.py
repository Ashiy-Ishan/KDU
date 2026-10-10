import uuid
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def utc_from_iso(iso_str: str) -> datetime:
    return datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

class Job:
    def __init__(self, job_id: str, tickets: list, model_version: str, idempotency_key: Optional[str] = None):
        self.job_id = job_id
        self.tickets = tickets
        self.total = len(tickets)
        self.processed = 0
        self.status = "queued"
        self.created_at = utc_now_iso()
        self.started_at = None
        self.finished_at = None
        self.expires_at = None
        self.model_version = model_version
        self.error = None
        self.predictions = []
        self.idempotency_key = idempotency_key
        self.cancelled_event = threading.Event()

    def to_status_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "expires_at": self.expires_at,
            "model_version": self.model_version,
            "error": self.error
        }

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        try:
            exp = utc_from_iso(self.expires_at)
            return datetime.now(timezone.utc) > exp
        except Exception:
            return False

class JobManager:
    MAX_ACTIVE_JOBS = 4
    RETENTION_HOURS = 6

    def __init__(self, predictor):
        self.predictor = predictor
        self.jobs: dict[str, Job] = {}
        self.idempotency_map: dict[str, str] = {}
        self.lock = threading.Lock()
        self.work_condition = threading.Condition(self.lock)
        self.stop_requested = False

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="JobWorker")
        self.worker_thread.start()

    def submit_job(self, tickets: list, idempotency_key: Optional[str] = None) -> tuple[int, dict, Optional[str]]:
        with self.lock:
            if idempotency_key and idempotency_key in self.idempotency_map:
                existing_id = self.idempotency_map[idempotency_key]
                if existing_id in self.jobs:
                    existing_job = self.jobs[existing_id]
                    if not existing_job.is_expired():
                        return 202, existing_job.to_status_dict(), existing_id

            active_count = sum(1 for j in self.jobs.values() if j.status in ("queued", "running"))
            if active_count >= self.MAX_ACTIVE_JOBS:
                return 429, {
                    "error": {
                        "code": "too_many_jobs",
                        "message": "Job queue is full. Retry later."
                    }
                }, None

            job_id = str(uuid.uuid4())
            model_ver = self.predictor.model_version or "v1.2"
            job = Job(job_id, tickets, model_ver, idempotency_key)
            self.jobs[job_id] = job

            if idempotency_key:
                self.idempotency_map[idempotency_key] = job_id

            self.work_condition.notify()
            return 202, job.to_status_dict(), job_id

    def get_job_status(self, job_id: str) -> tuple[int, dict]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return 404, {"error": {"code": "job_not_found", "message": "No job with this id."}}
            
            if job.is_expired():
                return 410, {"error": {"code": "job_expired", "message": "Job results have expired."}}

            return 200, job.to_status_dict()

    def get_job_results(self, job_id: str, offset: int = 0, limit: Optional[int] = None) -> tuple[int, dict]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return 404, {"error": {"code": "job_not_found", "message": "No job with this id."}}

            if job.is_expired():
                return 410, {"error": {"code": "job_expired", "message": "Job results have expired."}}

            if job.status != "succeeded":
                return 409, {
                    "error": {
                        "code": "job_not_ready",
                        "message": f"Job status is '{job.status}'. Results are available once it is 'succeeded'."
                    }
                }

            total = job.total
            if limit is None or limit <= 0:
                limit = total
            limit = min(limit, 5000)
            offset = max(0, offset)

            page = job.predictions[offset : offset + limit]
            next_offset = offset + len(page)
            if next_offset >= total:
                next_offset = None

            return 200, {
                "job_id": job.job_id,
                "status": "succeeded",
                "total": total,
                "offset": offset,
                "limit": limit,
                "next_offset": next_offset,
                "model_version": job.model_version,
                "predictions": page
            }

    def delete_job(self, job_id: str) -> tuple[int, Optional[dict]]:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return 404, {"error": {"code": "job_not_found", "message": "No job with this id."}}

            if job.status in ("queued", "running"):
                job.cancelled_event.set()
                job.status = "cancelled"
                job.finished_at = utc_now_iso()

            del self.jobs[job_id]
            if job.idempotency_key and job.idempotency_key in self.idempotency_map:
                del self.idempotency_map[job.idempotency_key]

            return 204, None

    def _worker_loop(self):
        while not self.stop_requested:
            target_job = None
            with self.work_condition:
                while not self.stop_requested:
                    queued_jobs = [j for j in self.jobs.values() if j.status == "queued"]
                    if queued_jobs:
                        queued_jobs.sort(key=lambda j: j.created_at)
                        target_job = queued_jobs[0]
                        target_job.status = "running"
                        target_job.started_at = utc_now_iso()
                        break
                    self.work_condition.wait(timeout=1.0)

            if target_job:
                self._process_job(target_job)

    def _process_job(self, job: Job):
        try:
            chunk_size = 100
            tickets = job.tickets
            all_preds = []

            for start_idx in range(0, len(tickets), chunk_size):
                if job.cancelled_event.is_set():
                    with self.lock:
                        job.status = "cancelled"
                        job.finished_at = utc_now_iso()
                    return

                chunk = tickets[start_idx : start_idx + chunk_size]
                chunk_preds = self.predictor.predict_batch(chunk)
                all_preds.extend(chunk_preds)

                with self.lock:
                    job.processed = len(all_preds)

            with self.lock:
                job.predictions = all_preds
                job.status = "succeeded"
                job.finished_at = utc_now_iso()
                now_dt = datetime.now(timezone.utc)
                exp_dt = now_dt + timedelta(hours=self.RETENTION_HOURS)
                job.expires_at = exp_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        except Exception as e:
            with self.lock:
                job.status = "failed"
                job.finished_at = utc_now_iso()
                now_dt = datetime.now(timezone.utc)
                exp_dt = now_dt + timedelta(hours=self.RETENTION_HOURS)
                job.expires_at = exp_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                job.error = {
                    "code": "internal_error",
                    "message": f"Processing failure: {str(e)}"
                }
