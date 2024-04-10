from benchita.task.ironita import IronIta

TASKS = dict(
    ironita=IronIta

)

def get_tasks():
    return list(TASKS.keys())

def get_task(name):
    if name not in TASKS:
        raise ValueError(f"Task {name} not found")

    return TASKS[name]