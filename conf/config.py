import pandas as pd


def to_boolean(x):
    if x == 'true':
        return True
    elif x == 'false':
        return False
    else:
        return x


def parser_abed_config(path_to_file):
    conf = pd.read_csv(path_to_file, header=None)
    conf.columns = ['key', 'value']
    conf['key'] = conf.key.map(lambda x: x.upper().replace('?', '').replace('--', '_').replace('-', '_'))
    conf['value'] = conf.value.map(lambda x: to_boolean(x))
    return {k: v for k, v in conf.values}
