from benchita.logging import log_info
from benchita.job import run_job

def worker(*, jobs_queue, results_queue, worker_id, device, args, experiment_name):
    while True:
        try:
            job = jobs_queue.get(False)
            log_info(f"Worker got a job... (remaing jobs = {jobs_queue.qsize()})", worker_id=worker_id)
            results = run_job(
                job=job,
                device=device,
                worker_id=worker_id,
                args=args,
                experiment_name=experiment_name
            )
            results_queue.put(results)
        except:
            log_info(f"No more jobs available, exiting...", worker_id=worker_id)
            break