class BaseWindow:
    _window_classes = {}
    def __init_subclass__(cls, window_type):
        cls._window_classes[window_type] = cls
        cls.window_type = window_type

    def to_dict(self):
        return {'type':self.window_type}

    @classmethod
    def from_dict(cls, d):
        # Subclasses must not call the parent implementation of this method!
        subclass_name = d['type']
        subclass = cls._window_classes[subclass_name]
        return subclass.from_dict(d)


class TopHatWindow(BaseWindow, window_type='TopHat'):
    def __init__(self, range_min, range_max):
        self.min = range_min
        self.max = range_max

    def to_dict(self):
        d = super().to_dict()
        d['min'] = self.min
        d['max'] = self.max
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d['min'], d['max'])


class Window(BaseWindow, window_type='Standard'):
    def __init__(self, values, weight):
        self.values = np.array(values)
        self.weight = np.array(weight)

    def to_dict(self):
        d = super().to_dict()
        d['values'] = self.values.tolist()
        d['weight'] = self.weight.tolist()

    @classmethod
    def from_dict(cls, d):
        return cls(d['values'], d['weight'])
