import numpy as np

# add more as we develop them
allowed_types = [
    "shear_xi_plus",
    "shear_xi_minus",
    "shear_ee",
    "shear_bb",
    "galaxy_density_cl",
    "galaxy_density_w",
    "ggl_gamma_t",
    "ggl_gamma_x",
    "ggl_cl",
]



class DataPoint:
    def __init__(self, data_type, tracers, value, **tags):
        self.data_type = data_type
        self.tracers = tracers
        self.value = value
        self.tags = tags
    
    def __repr__(self):
        return f"<Data {self.data_type} {self.tracers} {self.value} {self.tags}"
    
    def get_tag(self, tag):
        return self.tags.get(tag)
    

class Tracer:
    subclasses = {}
    def __init__(self, name):
        self.name = name

    def __init_subclass__(cls, tracer_type):
        cls.subclasses[tracer_type] = cls

    @classmethod
    def make(cls, tracer_type, name, *args, **kwargs):
        subclass = cls.subclasses[tracer_type]
        return subclass(name, *args, **kwargs)
        
class MiscTracer(Tracer, tracer_type='misc'):
    def __init__(self, name):
        super().__init__(name)

        
class NZTracer(Tracer, tracer_type='NZ'):
    def __init__(self, name, z, nz):
        super().__init__(name)
        self.name = name
        self.z = z
        self.nz = nz

# Define other tracer types as they come along

                
                

        
class Sacc:
    """
    A class containing a selection of LSST summary statistic measurements,
    their covariance, and the metadata necessary to compute theoretical
    predictions for them.
    """
    def __init__(self):
        self.data = []
        self.tracers = {}
        self.covariance = None
        self._mean = None

    def __len__(self):
        return len(self.data)

    def add_tracer(self, tracer_type, name, *args, **kwargs):
        """
        Add a new tracer
        """
        T = Tracer.make(tracer_type, name, *args, *kwargs)
        self.tracers[name] = T

    def add_data_point(self, data_type, tracers, value, **tags):
        """
        Add a new data point
        """
        if self.covariance is not None:
            raise ValueError("You cannot add a data point after setting the covariance")
        tracers = tuple(tracers)
        d = DataPoint(data_type, tracers, value, **tags)
        self.data.append(d)


    def cut(self, mask):
        """
        Remove data points and corresponding covariance elements following mask.

        Mask must be either a boolean array or a list of indices to remove.
        
        True = cut data point
        False = keep data point
        
        indices = data points to cut
        """
        mask = np.array(mask)
            
        if mask.dtype == np.bool:
            if not len(mask)==len(self):
                raise ValueError("Mask passed in is wrong size")
            self.data = [d for i,d in enumerate(self.data) if not mask[i]]
        else:
            # slow
            self.data = [d for i,d in enumerate(self.data) if not i in mask]
        print("Mask the covariance too!")

    def indices(self, data_type=None, tracers=None, **select):
        """
        Find the indices of all points matching the given selection
        """
        indices = []
        if tracers is not None:
            tracers = tuple(tracers)
        for i,d in enumerate(self.data):
            if not ((tracers is None) or (d.tracers == tracers)):
                continue
            if not ((data_type is None or d.data_type == data_type)):
                continue
            ok = True
            for name,val in select.items():
                if name.endswith("__lt"):
                    name = name[:-4]
                    if not d.get_tag(name) < val:
                        ok=False
                        break
                elif name.endswith("__gt"):
                    name = name[:-4]
                    if not d.get_tag(name) > val:
                        ok=False
                        break
                else:
                    if not d.get_tag(name) == val:
                        ok=False
                        break
                        
            if ok:
                indices.append(i)
        return np.array(indices)

    def get_tags(self, tags, data_type=None, tracers=None, **select):
        """
        Get the value of a one or more named tags for a subset of the data
        """
        indices = set(self.indices(data_type=data_type, tracers=tracers, **select))
        tags = [[d.get_tag(tag) for i,d in enumerate(self.data) if i in indices]
                for tag in tags]
        return tags
    
    def get_tag(self, tag, data_type=None, tracers=None, **select):
        """
        Get the value of a named tag for a subset of the data
        """
        return self.get_tags([tag], data_type=data_type, tracers=tracers, **select)[0]
    
    def get_data_points(self, data_type=None, tracers=None, **select):
        """
        Get data point objects for a subset of the data
        """
        indices = self.indices(data_type=data_type, tracers=tracers, **select)
        return [self.data[i] for i in indices]

    def get_mean(self, data_type=None, tracers=None, **select):
        """
        Get the vector of mean values for a selected subset of the data
        """
        indices = self.indices(data_type=data_type, tracers=tracers, **select)
        return self.mean[indices]

    def get_data_types(self):
        s = {d.data_type for d in self.data}
        return list(s)
    
    def get_tracer_combinations(self, data_type=None):
        """
        Get all sets of tracers used (e.g. tomographic bin pairs)
        """
        indices = self.indices(data_type=data_type)
        return list(set([self.data[i].tracers for i in indices]))
        

    @property
    def mean(self):
        """
        Get the vector of mean values for the entire data set.
        """
        if self._mean is None:
            self._mean = np.array([d.value for d in self.data])
        return self._mean

    @mean.setter
    def mean(self, mu):
        """
        Set the vector of mean values for the entire data set.
        """
        if not len(mu) == len(self.data):
            raise ValueError("Tried to set mean with thing of length {}"
                " but data is length {}".format(len(mu),len(self.data)))
        for m, d in zip(mu, self.data):
            d.value = m


    @classmethod
    def load(cls, filename):
        pass

    def save(self, filename):
        pass

    def add_covariance(self, covariance):
        pass
