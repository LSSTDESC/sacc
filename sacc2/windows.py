import numpy as np
from astropy.io import fits
from astropy.table import Table

class BaseWindow:
    _window_classes = {}
    def __init_subclass__(cls, window_type):
        cls._window_classes[window_type] = cls
        cls.window_type = window_type

    @classmethod
    def from_dict(cls, d):
        # Subclasses must not call the parent implementation of this method!
        subclass_name = d['type']
        subclass = cls._window_classes[subclass_name]
        return subclass.from_dict(d)

    @classmethod
    def to_fits(cls, instance_list):
        hdus = []
        for window_type, subclass in cls._window_classes.items():
            windows = [w for w in instance_list if type(w)==subclass]
            if not windows:
                continue
            data = [w.to_dict() for w in windows]
            tab = Table(data=data)
            hdu = fits.table_to_hdu(tab)
            hdu.name = str(window_type)
            hdu.header['saccclss'] = window_type
            hdu.header['sacctype'] = 'window'
            hdus.append(hdu)
        return hdus

    @classmethod
    def from_fits(cls, hdu):
        subclass_name = hdu.header['saccclss']
        subclass = cls._window_classes[subclass_name]
        windows= {row['id']: subclass.from_dict(row) for row in hdu.data}
        return windows





class TopHatWindow(BaseWindow, window_type='TopHat'):
    def __init__(self, range_min, range_max):
        self.min = range_min
        self.max = range_max

    def to_dict(self):
        d = {}
        d['min'] = self.min
        d['max'] = self.max
        d['id'] = id(self)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d['min'], d['max'])



class Window(BaseWindow, window_type='Standard'):
    def __init__(self, values, weight):
        self.values = np.array(values)
        self.weight = np.array(weight)

    def to_dict(self):
        d = {}
        d['values'] = self.values.tolist()
        d['weight'] = self.weight.tolist()
        d['id'] = id(self)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(d['values'], d['weight'])
