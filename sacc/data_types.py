import warnings
from collections import namedtuple
from astropy.table import Table

from .utils import (Namespace, hide_null_values,
                    null_values, camel_case_split_and_lowercase)

# The format for a data type name looks like this:
# {sources}_{properties}_{statistic_type}[_{statistic_subtype}]
#     sources: type(s) of astrophysical sources to which this applies
#     properties: feature(s)/characterisic(s) of those sources/fields to
#         which the statistic applies
#     statistic_type: mathematical type of the statistic
#     statistic_subtype: optional additional specifier

required_tags_concise = {
    "cl_00": ['ell'],
    "cl_0e": ['ell'],
    "cl_0b": ['ell'],
    "cl_e0": ['ell'],
    "cl_b0": ['ell'],
    "cl_ee": ['ell'],
    "cl_eb": ['ell'],
    "cl_be": ['ell'],
    "cl_bb": ['ell'],
    "xi_00": ['theta'],
    "xi_0e": ['theta'],
    "xi_0b": ['theta'],
    "xi_e0": ['theta'],
    "xi_b0": ['theta'],
    "xi_plus_re": ['theta'],
    "xi_plus_im": ['theta'],
    "xi_minus_re": ['theta'],
    "xi_minus_im": ['theta'],
    "count": []
}

required_tags_verbose = {
    "clusterGalaxy_densityConvergence_cl": ['ell'],
    "clusterGalaxy_densityConvergence_xi": ['theta'],
    "clusterGalaxy_densityShear_cl_b": ['ell'],
    "clusterGalaxy_densityShear_cl_e": ['ell'],
    "clusterGalaxy_densityShear_xi_t": ['theta'],
    "clusterGalaxy_densityShear_xi_x": ['theta'],
    "clusterGalaxy_density_cl": ['ell'],
    "clusterGalaxy_density_xi": ['theta'],
    "cluster_density_cl": ['ell'],
    "cluster_density_xi": ['theta'],
    "cluster_mass_count_wl": [],
    "cluster_mass_count_xray": [],
    "cmbCluster_polarizationDensity_cl_b": ['ell'],
    "cmbCluster_polarizationDensity_cl_e": ['ell'],
    "cmbCluster_polarizationDensity_xi_t": ['theta'],
    "cmbCluster_polarizationDensity_xi_x": ['theta'],
    "cmbCluster_temperatureDensity_cl": ['ell'],
    "cmbCluster_temperatureDensity_xi": ['theta'],
    "cmbCluster_convergenceDensity_cl": ['ell'],
    "cmbCluster_convergenceDensity_xi": ['theta'],
    "cmbGalaxy_convergenceDensity_cl": ['ell'],
    "cmbGalaxy_convergenceDensity_xi": ['theta'],
    "cmbGalaxy_convergenceShear_cl_b": ['ell'],
    "cmbGalaxy_convergenceShear_cl_e": ['ell'],
    "cmbGalaxy_convergenceShear_xi_t": ['theta'],
    "cmbGalaxy_convergenceShear_xi_x": ['theta'],
    "cmbGalaxy_convergence_cl": ['ell'],
    "cmbGalaxy_convergence_xi": ['theta'],
    "cmbGalaxy_polarizationConvergence_cl_b": ['ell'],
    "cmbGalaxy_polarizationConvergence_cl_e": ['ell'],
    "cmbGalaxy_polarizationConvergence_xi_t": ['theta'],
    "cmbGalaxy_polarizationConvergence_xi_x": ['theta'],
    "cmbGalaxy_polarizationDensity_cl_b": ['ell'],
    "cmbGalaxy_polarizationDensity_cl_e": ['ell'],
    "cmbGalaxy_polarizationDensity_xi_t": ['theta'],
    "cmbGalaxy_polarizationDensity_xi_x": ['theta'],
    "cmbGalaxy_polarizationShear_cl_bb": ['ell'],
    "cmbGalaxy_polarizationShear_cl_be": ['ell'],
    "cmbGalaxy_polarizationShear_cl_eb": ['ell'],
    "cmbGalaxy_polarizationShear_cl_ee": ['ell'],
    "cmbGalaxy_polarizationShear_xi_minus": ['theta'],
    "cmbGalaxy_polarizationShear_xi_plus": ['theta'],
    "cmbGalaxy_polarizationShear_xi_imagMinus": ['theta'],
    "cmbGalaxy_polarizationShear_xi_imagPlus": ['theta'],
    "cmbGalaxy_polarizationShear_cosebi_e": ['ell'],
    "cmbGalaxy_polarizationShear_cosebi_b": ['ell'],
    "cmbGalaxy_temperatureConvergence_cl": ['ell'],
    "cmbGalaxy_temperatureConvergence_xi": ['theta'],
    "cmbGalaxy_temperatureDensity_cl": ['ell'],
    "cmbGalaxy_temperatureDensity_xi": ['theta'],
    "cmbGalaxy_temperatureShear_cl_b": ['ell'],
    "cmbGalaxy_temperatureShear_cl_e": ['ell'],
    "cmbGalaxy_temperatureShear_xi_t": ['theta'],
    "cmbGalaxy_temperatureShear_xi_x": ['theta'],
    "cmb_convergence_cl": ['ell'],
    "cmb_convergence_xi": ['theta'],
    "cmb_polarization_cl_bb": ['ell'],
    "cmb_polarization_cl_be": ['ell'],
    "cmb_polarization_cl_eb": ['ell'],
    "cmb_polarization_cl_ee": ['ell'],
    "cmb_polarization_xi_minus": ['theta'],
    "cmb_polarization_xi_plus": ['theta'],
    "cmb_polarization_xi_imagMinus": ['theta'],
    "cmb_polarization_xi_imagPlus": ['theta'],
    "cmb_polarization_cosebi_e": [],
    "cmb_polarization_cosebi_b": [],
    "cmb_polarizationConvergence_cl_e": ['ell'],
    "cmb_polarizationConvergence_cl_b": ['ell'],
    "cmb_polarizationConvergence_xi_t": ['theta'],
    "cmb_polarizationConvergence_xi_x": ['theta'],
    "cmb_temperaturePolarization_cl_e": ['ell'],
    "cmb_temperaturePolarization_cl_b": ['ell'],
    "cmb_temperaturePolarization_xi_t": ['theta'],
    "cmb_temperaturePolarization_xi_x": ['theta'],
    "cmb_temperature_cl": ['ell'],
    "cmb_temperatureConvergence_cl": ["ell"],
    "cmb_temperatureConvergence_xi": ["theta"],
    "galaxy_convergenceDensity_cl": ['ell'],
    "galaxy_convergenceDensity_xi": ['theta'],
    "galaxy_convergence_cl": ['ell'],
    "galaxy_convergence_xi": ['theta'],
    "galaxy_density_cl": ['ell'],
    "galaxy_density_xi": ['theta'],
    "galaxy_shearConvergence_cl_b": ['ell'],
    "galaxy_shearConvergence_cl_e": ['ell'],
    "galaxy_shearConvergence_xi_t": ['theta'],
    "galaxy_shearConvergence_xi_x": ['theta'],
    "galaxy_shearDensity_cl_b": ['ell'],
    "galaxy_shearDensity_cl_e": ['ell'],
    "galaxy_shearDensity_xi_t": ['theta'],
    "galaxy_shearDensity_xi_x": ['theta'],
    "galaxy_shear_cl_bb": ['ell'],
    "galaxy_shear_cl_be": ['ell'],
    "galaxy_shear_cl_eb": ['ell'],
    "galaxy_shear_cl_ee": ['ell'],
    "galaxy_shear_cosebi_bb": [],
    "galaxy_shear_cosebi_ee": [],
    "galaxy_shear_xi_minus": ['theta'],
    "galaxy_shear_xi_plus": ['theta'],
    "galaxy_shear_xi_imagMinus": ['theta'],
    "galaxy_shear_xi_imagPlus": ['theta'],
}

required_tags = {**required_tags_concise, **required_tags_verbose}

parsedDataTypeName = namedtuple('parsedDataTypeName',
                                'sources properties statistic subtype')


def parse_data_type_name(name):
    """Parse a verbose data type name into its component parts
    Verbose data type names take the form:
    {sources}_{properties}_{statistic_type}[_{statistic_subtype}]
    where sources and properties are camel-case if there is more
    than one of them

    Parameters
    ----------
    name: str
        A data type name

    Returns
    -------
    sources: list[str]
        type(s) of astrophysical sources to which this applies

    properties: list[str]
        feature(s)/characterisic(s) of those sources/fields to
        which the statistic applies

    statistic_type: str
        mathematical type of the statistic

    statistic_subtype: str or None
        optional additional specifier
    """
    parts = name.split("_")
    if len(parts) == 3:
        sources, properties, statistic = parts
        subtype = None
    elif len(parts) == 4:
        sources, properties, statistic, subtype = parts
    else:
        raise ValueError("The supplied name is not a valid "
                         "verbose data type name (must have "
                         f"3 or 4 underscore-sparated parts): {name}")
    sources = camel_case_split_and_lowercase(sources)
    properties = camel_case_split_and_lowercase(properties)
    return parsedDataTypeName(sources, properties, statistic, subtype)


def build_data_type_name(sources, properties, statistic, subtype=None):
    """
    Parameters
    ----------
    sources: str or list[str]
        type(s) of astrophysical sources to which this applies

    properties: str or list[str]
        feature(s)/characterisic(s) of those sources/fields to
        which the statistic applies

    statistic_type: str
        mathematical type of the statistic

    statistic_subtype: str or None
        optional additional specifier.  Default is None

    Returns
    -------
    name: str
        Type name of the form:
           {sources}_{properties}_{statistic_type}[_{statistic_subtype}]
    """
    if not isinstance(sources, str):
        sources = "".join([sources[0]] + [s.lower().capitalize()
                                          for s in sources[1:]])
    if not isinstance(properties, str):
        properties = "".join([properties[0]] + [s.lower().capitalize()
                                                for s in properties[1:]])
    if subtype:
        return f"{sources}_{properties}_{statistic}_{subtype}"
    else:
        return f"{sources}_{properties}_{statistic}"


# This makes a namespace object, so you can do:
# standard_types.ggl_e == "ggl_e"
# also, for convenience, you can do standard_types.index('ggl_e')
# and 'ggl_e' in standard_types

standard_types = Namespace(*required_tags.keys())


class DataPoint:
    """A class for a single data point (one scalar value).

    Data points have a type, zero or more tracers, a value,
    and any arbitrary tags that are stored in a dictionary,
    and can be used to describe angular scales, window functions,
    or any arbitrary information to be attached to the data.

    Data points can be automatically created and added to a
    Sacc object, so you don't normally nee to manually create them.

    Parameters
    -----------
    data_type: str
        A string, indicating the type of data point

    tracers: tuple
        Tuple of strings with the names of tracers to use

    value: float
        Mean value of this statistics

    tags: dict
        Dictionary of further data point metadata, such as binning
        info, angles, etc.
    """
    def __init__(self, data_type, tracers, value,
                 ignore_missing_tags=False, **tags):
        """Create a new data point.

        Data points can be automatically created and added to a
        Sacc object, so you don't normally nee to manually create them.

        Parameters
        ----------
        data_type: str
            A string, indicating the type of data point

        tracers: tuple
            Tuple of strings with the names of tracers to use

        value: float
            Mean value of this statistics

        ignore_missing_tags: bool
            Optional, default=False.  If True, do not complain if a
            tracer usually needed for this data type is not present.

        **tags: dict[str:any]
            Dictionary of further data point metadata, such as binning
            info, angles, etc.
        """
        self.data_type = data_type
        self.tracers = tracers
        self.value = value
        self.tags = tags
        # Data types can have required tags which must be present.
        # Check for those here
        if (data_type in required_tags) and (not ignore_missing_tags):
            for tag in required_tags[data_type]:
                if tag not in tags:
                    raise ValueError("Tag {tag} required for data type "
                                     f"{data_type} "
                                     "(ignore_missing_tags=False)")

        # We encourage people to use existing type names, and issue a
        # warning if they do not to prod them in the right direction.
        if data_type not in standard_types:
            warnings.warn(f"Unknown data_type value {data_type}. "
                          "If possible use a pre-defined type, or "
                          "add to the list.")

    def __repr__(self):
        t = ", ".join(f'{k}={v}' for (k, v) in self.tags.items())
        st = f"DataPoint(data_type='{self.data_type}', "
        st += f"tracers={self.tracers}, value={self.value}, {t})"
        return st

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
    def to_table(cls, data, lookups=None):
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
        -------
        table: astropy.table.Table
            table object containing data points
        """
        if lookups is None:
            lookups = {}
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
    def from_table(cls, table, lookups=None):
        """Convert a table back into a list of data points.

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
        if lookups is None:
            lookups = {}
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
                # Now delete and null values, as indicated by the
                # sentinel above.
                if ((hasattr(tags[k], 'dtype')) and
                        (v == null_values[tags[k].dtype.kind])):
                    del tags[k]
            # Finally convert back to a data point and record
            data_point = cls(data_type, tracers, value, **tags)
            data.append(data_point)
        return data

    def _make_row(self, tracers, tags, lookups):
        """
        Turn this data point into a list with specified tracers and tags.
        If some tracers or tags are missing (homogenous data set) then
        use blank values or Nones for them.
        """
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
