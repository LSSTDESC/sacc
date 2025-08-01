from .base import BaseTracer, ONE_OBJECT_PER_TABLE, ONE_OBJECT_MULTIPLE_TABLES
from ..utils import remove_dict_null_values
from astropy.table import Table
import numpy as np

class MapTracer(BaseTracer, type_name='Map'):
    """
    A Tracer type for a sky map.

    Takes at least two arguments, defining the map beam.

    Parameters
    ----------
    name: str
        The name for this specific tracer object.
    ell: array
         Array of multipole values at which the beam is defined.
    beam: array
         Beam multipoles at each value of ell.
    beam_extra: array
         Other beam-related arrays
         (e.g. uncertainties, principal components,
         alternative measurements, whatever).
    map_unit: str
         Map units (e.g. 'uK_CMB'). 'none' by default.
    """

    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(self, name, spin, ell, beam,
                 beam_extra=None, map_unit='none', **kwargs):
        super().__init__(name, **kwargs)
        self.spin = spin
        self.map_unit = map_unit
        self.ell = np.array(ell)
        self.beam = np.array(beam)
        self.beam_extra = {} if beam_extra is None else beam_extra

    def to_table(self):
        # Beams
        names = ['ell', 'beam']
        cols = [self.ell, self.beam]
        for beam_id, col in self.beam_extra.items():
            names.append(str(beam_id))
            cols.append(col)
        table = Table(data=cols, names=names)
        table.meta['SACCQTTY'] = self.quantity
        table.meta['SACCNAME'] = self.name
        extname = f'tracer:{self.type_name}:{self.name}:beam'
        table.meta['EXTNAME'] = extname
        table.meta['MAP_UNIT'] = self.map_unit
        table.meta['SPIN'] = self.spin
        for key, value in self.metadata.items():
            table.meta['META_'+key] = value
        remove_dict_null_values(table.meta)

        return table


    @classmethod
    def from_table(cls, table):
        """Convert a single astropy table into a MapTracer instance.

        This is used when loading data from a file.

        Parameters
        ----------
        table: astropy.table.Table

        Returns
        -------
        tracer: MapTracer
            An instance of MapTracer created from the table.
        """
        name = table.meta['SACCNAME']
        quantity = table.meta.get('SACCQTTY', 'generic')
        map_unit = table.meta['MAP_UNIT']
        spin = table.meta['SPIN']
        metadata = {key[5:]: value for key, value in table.meta.items() if key.startswith("META_")}

        ell = table['ell']
        beam = table['beam']
        beam_extra = {col.name: col.data for col in table.columns.values() if col.name not in ['ell', 'beam']}

        return cls(name, spin, ell, beam, beam_extra=beam_extra, map_unit=map_unit, quantity=quantity, metadata=metadata)


class NuMapTracer(BaseTracer, type_name='NuMap'):
    """
    A Tracer type for a sky map at a given frequency.

    Takes at least four arguments, defining the bandpass and beam.

    Parameters
    ----------
    name: str
        The name for this specific tracer, e.g. a frequency band
        identifier.
    spin: int
        Spin for this observable. Either 0 (e.g. intensity)
        or 2 (e.g. polarization).
    nu: array
         Array of frequencies.
    bandpass: array
         Bandpass transmission.
    bandpass_extra: array
         Other bandpass-related arrays
         (e.g. uncertainties, principal components,
         alternative measurements, whatever).
    ell: array
         Array of multipole values at which the beam is defined.
    beam: array
         Beam.
    beam_extra: array
         Other beam-related arrays
         (e.g. uncertainties, principal components,
         alternative measurements, whatever).
    nu_unit: str
         Frequency units ('GHz' by default).
    map_unit: str
         Map units (e.g. 'uK_CMB'). 'none' by default.
    """

    storage_type = ONE_OBJECT_MULTIPLE_TABLES

    def __init__(self, name, spin, nu, bandpass,
                 ell, beam, bandpass_extra=None,
                 beam_extra=None, nu_unit='GHz',
                 map_unit='none', **kwargs):
        super().__init__(name, **kwargs)
        self.spin = spin
        self.nu = np.array(nu)
        self.nu_unit = nu_unit
        self.map_unit = map_unit
        self.bandpass = np.array(bandpass)
        self.bandpass_extra = {} if bandpass_extra is None else bandpass_extra
        self.ell = np.array(ell)
        self.beam = np.array(beam)
        self.beam_extra = {} if beam_extra is None else beam_extra

    def to_tables(self):
        # Bandpass
        names = ['nu', 'bandpass']
        cols = [self.nu, self.bandpass]
        for bandpass_id, col in self.bandpass_extra.items():
            names.append(str(bandpass_id))
            cols.append(col)
        bandpass_table = Table(data=cols, names=names)
        bandpass_table.meta['SACCQTTY'] = self.quantity
        bandpass_table.meta['NU_UNIT'] = self.nu_unit
        bandpass_table.meta['SACCNAME'] = self.name
        bandpass_table.meta['SPIN'] = self.spin
        for key, value in self.metadata.items():
            bandpass_table.meta['META_'+key] = value
        remove_dict_null_values(bandpass_table.meta)

        # Beam
        names = ['ell', 'beam']
        cols = [self.ell, self.beam]
        for beam_id, col in self.beam_extra.items():
            names.append(str(beam_id))
            cols.append(col)
        beam_table = Table(data=cols, names=names)
        beam_table.meta['SACCQTTY'] = self.quantity
        beam_table.meta['MAP_UNIT'] = self.map_unit
        beam_table.meta['SPIN'] = self.spin
        for key, value in self.metadata.items():
            beam_table.meta['META_'+key] = value
        remove_dict_null_values(beam_table.meta)

        return {'bandpass': bandpass_table, 'beam': beam_table}

    @classmethod
    def from_tables(cls, table_dict):
        """Convert a dictionary of astropy tables into a NuMapTracer instance."""
        bandpass_table = table_dict['bandpass']
        beam_table = table_dict['beam']

        # Get the various bits of metadata out of the bandpass table
        name = bandpass_table.meta['SACCNAME']
        spin = bandpass_table.meta['SPIN']
        quantity = bandpass_table.meta.get('SACCQTTY', 'generic')
        nu_unit = bandpass_table.meta['NU_UNIT']

        # Additional miscellaneous metadata
        metadata = {key[5:]: value for key, value in bandpass_table.meta.items() if key.startswith("META_")}

        # And the actual bandpass data columns themselves
        nu = bandpass_table['nu']
        bandpass = bandpass_table['bandpass']
        bandpass_extra = {col.name: col.data for col in bandpass_table.columns.values() if col.name not in ['nu', 'bandpass']}

        # Now the same for the beam table
        ell = beam_table['ell']
        beam = beam_table['beam']
        beam_extra = {col.name: col.data for col in beam_table.columns.values() if col.name not in ['ell', 'beam']}
        map_unit = beam_table.meta['MAP_UNIT']

        return cls(name, spin, nu, bandpass,
                   ell, beam,
                   bandpass_extra=bandpass_extra,
                   beam_extra=beam_extra,
                   nu_unit=nu_unit,
                   map_unit=map_unit,
                   quantity=quantity,
                   metadata=metadata)
