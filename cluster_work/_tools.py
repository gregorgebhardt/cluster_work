import collections
import re


def format_time(time_in_secs: float):
    _hours = int(time_in_secs) // 60 ** 2
    _minutes = (int(time_in_secs) // 60) % 60
    _seconds = time_in_secs % 60

    time_str = ""
    if _hours:
        time_str += "{:d}h:".format(_hours)
    if _minutes or _hours:
        time_str += "{:d}m:".format(_minutes)
    time_str += "{:05.2f}s".format(_seconds)

    return time_str


def format_counter(counter: int) -> int:
    return counter + 1


def de_camelcase_underscore(string: str):
    str_parts = re.findall('[A-Z][^A-Z_]*|[^A-Z_]+', string)
    return ''.join([s[0] for s in str_parts])


def shorten_param(_param_name):
    name_parts = _param_name.split('.')
    shortened_parts = '.'.join(map(de_camelcase_underscore, name_parts[:-1]))
    shortened_leaf = ''.join(map(lambda s: s[0], name_parts[-1].split('_')))
    if shortened_parts:
        return shortened_parts + '.' + shortened_leaf
    else:
        return shortened_leaf

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = deep_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, collections.MutableSequence):
            keys = map(lambda i: new_key + "_" + str(i), range(len(v)))
            items.extend(zip(keys, v))
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_dict_to_tuple_keys(d: collections.MutableMapping):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, collections.MutableMapping):
            sub_dict = flatten_dict_to_tuple_keys(v)
            flat_dict.update({(k, *sk): sv for sk, sv in sub_dict.items()})

        elif isinstance(v, collections.MutableSequence):
            flat_dict[(k,)] = v

    return flat_dict


def insert_deep_dictionary(d: collections.MutableMapping, t: tuple, value):
    if type(t) is tuple:
        if len(t) == 1:  # tuple contains only one key
            d[t[0]] = value
        else:  # tuple contains more than one key
            if t[0] not in d:
                d[t[0]] = dict()
            insert_deep_dictionary(d[t[0]], t[1:], value)
    else:
        d[t] = value
