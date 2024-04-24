import os
import json
import hashlib
import traceback

from benchita.logging import log_info, log_error
from benchita.evaluate import evaluate

def get_job_name(job):
    job_dict = dict(task=job["task"].model_dump(), model=job["model"].model_dump())
    if job_dict["model"]["peft"] is None: del job_dict["model"]["peft"]  # correctly reuse pre-peft cached results
    job_config = json.dumps(job_dict, sort_keys=True)
    hash = hashlib.sha256(job_config.encode("utf-8")).hexdigest()
    name = f"{job['model'].model.name}-{job['task'].name}-{hash[:16]}"
    name = name.replace("/", "_")
    return name

def run_job(*, job, device, args, experiment_name, worker_id):
    try:
        job_name = get_job_name(job)
        results_path = os.path.join(args.results_dir, experiment_name)
        os.makedirs(results_path, exist_ok=True)
        results_file = os.path.join(results_path, f"{job_name}.json")
        if os.path.exists(results_file):
            log_info(f"Skipping job {job_name}, results already exist")
            with open(results_file, "r") as f:
                results = json.load(f)
            return results
        else:
            log_info(f"Running job {job_name}")
            return evaluate(job=job, device=device, args=args, results_file=results_file, worker_id=worker_id)
    except Exception as e:
        traceback_details = traceback.format_exc()
        error_message = f"An error occurred while processing job {job_name}: {str(e)}\nTraceback for the error: {traceback_details}"
        log_error(error_message, worker_id=worker_id)