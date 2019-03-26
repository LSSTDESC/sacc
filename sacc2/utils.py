from collections import OrderedDict



def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]



class Namespace(OrderedDict):
    def __init__(self, strings):
        for s in strings:
            self.__dict__[s] = s
