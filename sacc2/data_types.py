import warnings

from astropy.table import Table

from .utils import Namespace, hide_null_values, null_values

# This makes a namespace object, so you can do:
# known_types.ggl_E == "ggl_E"
# also, for convenience, you can do known_types.index('ggl_E') 
# and 'ggl_E' in known_types
known_types = Namespace(
    "galaxy_shear_xi_plus",
    "galaxy_shear_xi_minus",
    "galaxy_shear_xi_plus_imaginary",
    "galaxy_shear_xi_minus_imaginary",
    "galaxy_shear_ee",
    "galaxy_shear_bb",
    "galaxy_shear_eb",
    "galaxy_density_cl",
    "galaxy_density_w",
    "ggl_gamma_t",
    "ggl_gamma_x",
    "ggl_E",
    "ggl_B",
)

required_tags = {
    "galaxy_shear_xi_plus": ["theta"],
    "galaxy_shear_xi_minus": ["theta"],
    "galaxy_shear_xi_plus_imaginary": ["theta"],
    "galaxy_shear_xi_minus_imaginary": ["theta"],
    "galaxy_density_w":["theta"] ,
    "ggl_gamma_t":["theta"] ,
    "ggl_gamma_x":["theta"] ,
    "galaxy_shear_ee": ["ell"],
    "galaxy_shear_bb": ["ell"],
    "galaxy_shear_eb": ["ell"],
    "galaxy_density_cl": ["ell"],
    "ggl_E": ["ell"],
    "ggl_B": ["ell"],
    

}

class DataPoint:
    def __init__(self, data_type, tracers, value, ignore_missing_tags=False, **tags):
        self.data_type = data_type
        self.tracers = tracers
        self.value = value
        self.tags = tags
        # Data types can have required tags which must be present.
        # Check for those here
        if (data_type in required_tags) and (not ignore_missing_tags):
            for tag in required_tags[data_type]:
                if tag not in tags:
                    raise ValueError(f"Tag {tag} required for data type {data_type} (ignore_missing_tags=False)")
        # We encourage people to use existing type names, and issue a warning if they do
        # not to prod them in the right direction.
        if data_type not in known_types:
            warnings.warn(f"Unknown data_type value {data_type}. If possible use a pre-defined type, or add to the list.")

    def __repr__(self):
        return f"<Data {self.data_type} {self.tracers} {self.value} {self.tags}>s"

    def get_tag(self, tag, default=None):
        """
        Get the value of the the named tag, or None if not found.

        Parameters
        ----------
        tag: str
            Tag to find on the data point
        default: any
            Value to return if the tag is not found

        Returns
        -------
        value: any
            Value of the tag in this data point
        """
        return self.tags.get(tag, default)

    def __getitem__(self, tag):
        """
        Get the value of the the named tag, raising an
        error if it is not found

        Parameters
        ----------
        tag: str
            Tag to find on the data point

        Returns
        -------
        value: any
            Value of the tag in this data point
        """
        return self.tags[tag]

    @staticmethod
    def _choose_fields(data):
        """
        Internal static method to generate a list of colum names from a list
        of data points.  Since the data points can be heterogenous then this
        is not quite trivial - we use the union of the tag names and tracer_0,
        tracer_1, etc. up to the max number of tracers.
        """
        tags = set()
        ntracer = 0
        for d in data:
            ntracer = max(ntracer, len(d.tracers))
            tags.update(d.tags.keys())
        tags = list(tags)
        tracers = [f'tracer_{i}' for i in range(ntracer)]
        return tracers, tags

    @classmethod
    def to_table(cls, data, lookups={}):
        """
        Convert a list of data points to a single homogenous table

        Since data points can have varying tags, this method uses
        null values to represent non-present tags.

        Parameters
        ----------

        data: list
            A list of DataPoint objects

        lookups: dict
            A dictionary of tags->dict showing replacements to make
            in the tags. Default is empty.

        Returns

        table: astropy.table.Table
            table object containing data points
        """
        # Get the names of the columns to generate
        tracers, tags = cls._choose_fields(data)
        names = tracers + ['value'] + tags
        ntracer = len(tracers)
        # Convert each data point to a row
        rows = [d._make_row(tracers, tags, lookups) for d in data]

        # Convert to a table and fiddle slightly.
        table = Table(rows=rows, names=names)
        table.meta['NTRACER'] = ntracer
        hide_null_values(table)
        return table

    @classmethod
    def from_table(cls, table, lookups={}):
        """
        Convert a table back into a list of data points.

        This method removes null values from the tags.

        Parameters
        ----------

        table: astropy.table.Table
            A table of data containing the tracers, values, and tags

        lookups: dict
            A dictionary of tags->dict showing replacements to make
            in the tags. Default is empty.

        Returns
        -------

        data: list
            list of DataPoint objects
        """
        # Get out required table metadata
        nt = table.meta['NTRACER']
        data_type = table.meta['SACCNAME']

        # Tag names - we will remove missing tags below
        tag_names = table.colnames[nt + 1:]
        data = []
        for row in table:
            # Get basic data elements
            tracers = tuple([row[f'tracer_{i}'] for i in range(nt)])
            value = row['value']

            # Deal with tags.  First just pull out all remaining columns
            tags = {name: row[name] for name in tag_names}
            for k, v in list(tags.items()):
                # Deal with any tags that we should replace.
                # This is mainly used for Window instances.
                if k in lookups:
                    tags[k] = lookups[k].get(v, v)
                # Now delete and null values, as indicated by the sentinel above.
                if hasattr(tags[k], 'dtype') and v == null_values[tags[k].dtype.kind]:
                    del tags[k]
            # Finally convert back to a data point and record
            data_point = cls(data_type, tracers, value, **tags)
            data.append(data_point)
        return data

    def _make_row(self, tracers, tags, lookups):
        # Turn this data point into a list with specified tracers and tags.
        # If some tracers or tags are missing (homogenous data set) then
        # use blank values or Nones for them.
        nt = len(tracers)
        missing = nt - len(self.tracers)
        row = list(self.tracers) + ["" for i in range(missing)]
        row.append(self.value)
        for t in tags:
            v = self.tags.get(t)
            lookup = lookups.get(t)
            if lookup is not None:
                v = lookup.get(v, v)
            row.append(v)
        return row
