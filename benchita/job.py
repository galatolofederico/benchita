import os
import json
import hashlib

from benchita.logging import log_info
from benchita.evaluate import evaluate

def get_job_name(job):
    job_config = json.dumps(dict(task=job["task"].model_dump(), model=job["model"].model_dump()), sort_keys=True)
    hash = hashlib.sha256(job_config.encode("utf-8")).hexdigest()
    name = f"{job['model'].model.name}-{job['task'].name}-{hash[:16]}"
    name = name.replace("/", "_")
    return name

def run_job(*, job, args, experiment_name):
    job_name = get_job_name(job)
    results_path = os.path.join(args.results_dir, experiment_name)
    os.makedirs(results_path, exist_ok=True)
    results_file = os.path.join(results_path, f"{job_name}.json")
    if os.path.exists(results_file):
        log_info(f"Skipping job {job_name}, results already exist")
        return
    else:
        log_info(f"Running job {job_name}")
        evaluate(job=job, args=args, results_file=results_file)