from termcolor import colored
import sys

def log_info(message):
    print(colored("[INFO] "+message, "white"))

def log_warn(message):
    print(colored("[WARN] "+message, "yellow"))

def log_error(message, die=True):
    print(colored("[ERROR] "+message, "red"))
    if die: sys.exit(1)