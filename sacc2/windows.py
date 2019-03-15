

class BaseWindow:
    pass



class TopHatWindow(BaseWindow):
    def __init__(self, range_min, range_max):
        self.min = range_min
        self.max = range_max

class Window(BaseWindow):
    def __init__(self, values, weight):
        self.values = values
        self.weight = weight

