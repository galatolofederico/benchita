import argparse
import torch

from benchita.config import parse_config
from benchita.utils import build_jobs
from benchita.logging import log_info
from benchita.job import run_job

def main():
    parser = argparse.ArgumentParser(description="benchita")
    parser.add_argument("cmd", type=str, nargs="?", help="Command to execute", choices=["evaluate", "collect"])
    
    parser.add_argument("--config", type=str, default="config.yaml", help="The configuration file to use")
    parser.add_argument("--dry-run", action="store_true", help="Dry run the task")
    parser.add_argument("--dummy-run", action="store_true", help="Run the task using a dummy model")
    parser.add_argument("--results-dir", type=str, default="./results", help="The directory to save the results")
    parser.add_argument("--devices", type=int, default=None, nargs="*", help="The devices to use")
    
    args = parser.parse_args()

    config = parse_config(args.config)
    jobs = build_jobs(config)
    log_info(f"{len(jobs)} jobs to run")

    if args.devices is None:
        args.devices = list(range(torch.cuda.device_count()))

    log_info(f"Running jobs in parallel using {len(args.devices)} devices")

    run_job(job=jobs[0], args=args, experiment_name=config.experiment)

if __name__ == "__main__":
    main()