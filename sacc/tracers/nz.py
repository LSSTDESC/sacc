from .base import BaseTracer, ONE_OBJECT_PER_TABLE, ONE_OBJECT_MULTIPLE_TABLES
from astropy.table import Table
import numpy as np
from ..utils import remove_dict_null_values

class NZTracer(BaseTracer, type_name='NZ'):
    """
    A Tracer type for tomographic n(z) data.

    Takes two arguments arrays of z and n(z)

    Parameters
    ----------
    name: str
        The name for this specific tracer, e.g. a
        tomographic bin identifier.

    z: array
        Redshift sample values

    nz: array
        Number density n(z) at redshift sample points.

    extra_columns: dict[str: array] or dict[int: array]
        Additional estimates of the same n(z), by name
    """

    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(self, name, z, nz,
                 extra_columns=None, **kwargs):
        """
        Create a tracer corresponding to a distribution in redshift n(z),
        for example of galaxies.

        Parameters
        ----------
        name: str
            The name for this specific tracer, e.g. a
            tomographic bin identifier.

        z: array
            Redshift sample values

        nz: array
            Number density n(z) at redshift sample points.

        extra_columns: dict[str:array]
            Optional, default=None.  Additional realizations or
            estimates of the same n(z), by name.

        Returns
        -------
        instance: NZTracer object
            An instance of this class
        """
        super().__init__(name, **kwargs)
        self.z = np.array(z)
        self.nz = np.array(nz)
        self.extra_columns = {} if extra_columns is None else extra_columns

    def to_table(self):
        """Convert a list of NZTracers to a list of astropy tables

        This is used when saving data to a file.
        One table is generated per tracer.

        Parameters
        ----------
        instance_list: list
            List of tracer instances

        Returns
        -------
        tables: list
            List of astropy tables
        """
        names = ['z', 'nz']
        cols = [self.z, self.nz]
        for nz_id, col in self.extra_columns.items():
            names.append(str(nz_id))
            cols.append(col)
        table = Table(data=cols, names=names)
        table.meta['SACCQTTY'] = self.quantity
        # This will also get set at the higher level
        # if the tracer is save to a file, but for
        # testing it's useful to have it here too.
        table.meta['SACCNAME'] = self.name
        for key, value in self.metadata.items():
            table.meta['META_'+key] = value
        remove_dict_null_values(table.meta)
        return table

    @classmethod
    def from_table(cls, table):
        """Convert an astropy table into a an n(z) tracer

        This is used when loading data from a file.

        Parameters
        ----------
        table: astropy.table.Table
            Must contain the appropriate data, for example as saved
            by to_table.

        Returns
        -------
        tracers: dict
            Dict mapping string names to tracer objects.
            Only contains one key/value pair for the one tracer.
        """
        name = table.meta['SACCNAME']
        quantity = table.meta.get('SACCQTTY', 'generic')
        z = table['z']
        nz = table['nz']
        extra_columns = {}
        for col in table.columns.values():
            if col.name not in ['z', 'nz']:
                extra_columns[col.name] = col.data

        metadata = {}
        for key, value in table.meta.items():
            if key.startswith("META_"):
                metadata[key[5:]] = value
        return cls(name, z, nz,
                            quantity=quantity,
                            extra_columns=extra_columns,
                            metadata=metadata)



class QPNZTracer(BaseTracer, type_name='QPNZ'):
    """
    A Tracer type for tomographic n(z) data represented as a `qp.Ensemble`

    Takes a `qp.Ensemble` and optionally a redshift array.

    Requires the `qp` and `tables_io` packages to be installed.

    Parameters
    ----------
    name: str
        The name for this specific tracer, e.g. a
        tomographic bin identifier.

    ensemble: qp.Ensemble
        The qp.ensemble in questions
    """

    storage_type = ONE_OBJECT_MULTIPLE_TABLES

    def __init__(self, name, ens, z=None, **kwargs):
        """
        Create a tracer corresponding to a distribution in redshift n(z),
        for example of galaxies.

        Parameters
        ----------
        name: str
            The name for this specific tracer, e.g. a
            tomographic bin identifier.

        ensemble: qp.Ensemble
            The qp.ensemble in questions

        z: array
            Optional grid of redshift values at which to evaluate the ensemble.
            If left as None then the ensemble metadata is checked for a grid.
            If that is not present then no redshift grid is saved.

        Returns
        -------
        instance: NZTracer object
            An instance of this class
        """
        super().__init__(name, **kwargs)
        self.ensemble = ens
        if z is None:
            ens_meta = ens.metadata()
            if 'bins' in list(ens_meta.keys()):
                z = ens_meta['bins'][0]
        self.z = z
        if z is None:
            self.nz = None
        else:
            self.nz = np.mean(ens.pdf(self.z),axis=0)
        
    def to_tables(self):
        """Convert a list of NZTracers to a list of astropy tables

        This is used when saving data to a file.
        Two or three tables are generated per tracer.

        Parameters
        ----------
        instance_list: list
            List of tracer instances

        Returns
        -------
        tables: list
            List of astropy tables
        """
        from tables_io.convUtils import convertToApTables

        tables = {}

        if self.z is not None:
            names = ['z', 'nz']
            cols = [self.z, self.nz]
            fid_table = Table(data=cols, names=names)
            fid_table.meta['SACCQTTY'] = self.quantity

        table_dict = self.ensemble.build_tables()
        ap_tables = convertToApTables(table_dict)
        data_table = ap_tables['data']
        meta_table = ap_tables['meta']
        ancil_table = ap_tables.get('ancil', None)
        meta_table.meta['SACCQTTY'] = self.quantity
        data_table.meta['SACCQTTY'] = self.quantity

        for kk, vv in self.metadata.items():
            meta_table.meta['META_'+kk] = vv
        tables.append(data_table)

        tables.append(meta_table)
        if self.z is not None:
            tables.append(fid_table)
        if ancil_table:
            ancil_table.meta['SACCQTTY'] = self.quantity

        return tables

    @classmethod
    def from_tables(cls, table_list):
        """Convert an astropy table into a dictionary of tracers

        This is used when loading data from a file.
        A single tracer object is read from the table.

        Parameters
        ----------
        table_list: list[astropy.table.Table]
            Must contain the appropriate data, for example as saved
            by to_table.

        Returns
        -------
        tracers: dict
            Dict mapping string names to tracer objects.
            Only contains one key/value pair for the one tracer.
        """
        import qp

        tracers = {}
        sorted_dict = {}
        for table_ in table_list:
            tokens = table_.meta['EXTNAME'].split(':')
            table_key = f'{tokens[0]}:{tokens[1]}:{tokens[2]}'
            table_type = f'{tokens[3]}'
            if table_key not in sorted_dict:
                sorted_dict[table_key] = {table_type: table_}
            else:
                sorted_dict[table_key][table_type] = table_

        for val in sorted_dict.values():
            meta_table = val['meta']
            if 'fid' in val:
                z = val['fid']['z']
            else:
                z = None
            ensemble = qp.from_tables(val)
            name = meta_table.meta['SACCNAME']
            quantity = meta_table.meta.get('SACCQTTY', 'generic')
            ensemble = qp.from_tables(val)
            metadata = {}
            for key, value in meta_table.meta.items():
                if key.startswith("META_"):
                    metadata[key[5:]] = value
            tracers[name] = cls(name, ensemble, z=z,
                                quantity=quantity,
                                metadata=metadata)
        return tracers
