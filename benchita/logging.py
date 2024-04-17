from termcolor import colored
import sys

def log_info(message, worker_id=-1):
    worker = f" [worker {worker_id}]" if worker_id >= 0 else " [main]"
    print(colored(f"[INFO]{worker} {message}", "white"))

def log_warn(message, worker_id=-1):
    worker = f" [worker {worker_id}]" if worker_id >= 0 else " [main]"
    print(colored(f"[WARN]{worker} {message}", "yellow"))

def log_error(message, die=True, worker_id=-1):
    worker = f" [worker {worker_id}]" if worker_id >= 0 else " [main]"
    print(colored(f"[ERROR]{worker} {message}", "red"))
    if die: sys.exit(1)