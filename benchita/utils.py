

def parse_srt_args(s):
    ret = dict()
    for k, v in [arg.split("=") for arg in s.split(",")]:
        ret[k] = v
    return ret