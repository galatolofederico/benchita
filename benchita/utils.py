

def parse_str_args(s):
    ret = dict()
    for kv in [arg.split("=") for arg in s.split(",")]:
        if len(kv) == 2:
            k, v = kv
            ret[k] = v
    return ret