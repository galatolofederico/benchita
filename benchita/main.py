import argparse
import torch
import pandas as pd
from multiprocessing import Queue, Process
import io

from benchita.config import parse_config
from benchita.utils import build_jobs
from benchita.logging import log_info
from benchita.worker import worker

def main():
    parser = argparse.ArgumentParser(description="benchita")
    parser.add_argument("cmd", type=str, nargs="?", help="Command to execute", choices=["evaluate", "collect"])
    
    parser.add_argument("--config", type=str, default="config.yaml", help="The configuration file to use")
    parser.add_argument("--dry-run", action="store_true", help="Dry run the task")
    parser.add_argument("--dummy-run", action="store_true", help="Run the task using a dummy model")
    parser.add_argument("--results-dir", type=str, default="./results", help="The directory to save the results")
    parser.add_argument("--devices", type=int, default=None, nargs="*", help="The devices to use")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use when --no-parallel")
    parser.add_argument("--no-parallel", action="store_true", help="Do not use parallelism")
    
    args = parser.parse_args()

    config = parse_config(args.config)
    jobs = build_jobs(config)
    log_info(f"{len(jobs)} jobs to run")

    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))

    jobs_queue = Queue()
    results_queue = Queue()

    for job in jobs: jobs_queue.put_nowait(job)


    if args.no_parallel:
        for job in jobs:
            worker(
                jobs_queue=jobs_queue,
                results_queue=results_queue,
                device=torch.device(args.device),
                worker_id=-1,
                args=args,
                experiment_name=config.experiment
            )
    else:
        log_info(f"Running jobs in parallel using {len(args.devices)} devices")

        workers = []
        for i, device in enumerate(args.devices):
            device = torch.cuda.device(device)
            workers.append(Process(
                target=worker,
                kwargs=dict(
                    jobs_queue=jobs_queue,
                    results_queue=results_queue,
                    device=device,
                    worker_id=i,
                    args=args,
                    experiment_name=config.experiment
                )
            ))

        for _worker in workers: _worker.start()
        for _worker in workers: _worker.join()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    summary = pd.concat([pd.read_json(io.StringIO(result["summary"])) for result in results])
    log_info("Final results:")
    print(summary)

if __name__ == "__main__":
    main()