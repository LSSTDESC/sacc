from .base import BaseTracer
from ..utils import remove_dict_null_values
from astropy.table import Table
import numpy as np

class MapTracer(BaseTracer, tracer_type='Map'):
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

    def __init__(self, name, spin, ell, beam,
                 beam_extra=None, map_unit='none', **kwargs):
        super().__init__(name, **kwargs)
        self.spin = spin
        self.map_unit = map_unit
        self.ell = np.array(ell)
        self.beam = np.array(beam)
        self.beam_extra = {} if beam_extra is None else beam_extra

    @classmethod
    def to_tables(cls, instance_list):
        tables = []
        for tracer in instance_list:
            # Beams
            names = ['ell', 'beam']
            cols = [tracer.ell, tracer.beam]
            for beam_id, col in tracer.beam_extra.items():
                names.append(str(beam_id))
                cols.append(col)
            table = Table(data=cols, names=names)
            table.meta['SACCTYPE'] = 'tracer'
            table.meta['SACCCLSS'] = cls.tracer_type
            table.meta['SACCNAME'] = tracer.name
            table.meta['SACCQTTY'] = tracer.quantity
            extname = f'tracer:{cls.tracer_type}:{tracer.name}:beam'
            table.meta['EXTNAME'] = extname
            table.meta['MAP_UNIT'] = tracer.map_unit
            table.meta['SPIN'] = tracer.spin
            for key, value in tracer.metadata.items():
                table.meta['META_'+key] = value
            remove_dict_null_values(table.meta)
            tables.append(table)
        return tables

    @classmethod
    def from_tables(cls, table_list):
        tracers = {}

        # Collect beam and bandpass tables describing the same tracer
        tr_tables = {}
        for table in table_list:
            # Read name and table type
            name = table.meta['SACCNAME']
            tabtyp = table.meta['EXTNAME'].split(':')[-1]
            if tabtyp not in ['beam']:
                raise KeyError("Unknown table type " + table.meta['EXTNAME'])

            # If not present yet, create new tracer entry
            if name not in tr_tables:
                tr_tables[name] = {}
            # Add table
            tr_tables[name][tabtyp] = table

        # Now loop through different tracers and build them from their tables
        for dt in tr_tables.values():
            quantity = []
            metadata = {}
            map_unit = 'none'
            ell = []
            beam = []
            beam_extra = {}
            spin = 0

            if 'beam' in dt:
                table = dt['beam']
                name = table.meta['SACCNAME']
                quantity = table.meta.get('SACCQTTY', 'generic')
                ell = table['ell']
                beam = table['beam']
                for col in table.columns.values():
                    if col.name not in ['ell', 'beam']:
                        beam_extra[col.name] = col.data
                map_unit = table.meta['MAP_UNIT']
                spin = table.meta['SPIN']
                for key, value in table.meta.items():
                    if key.startswith("META_"):
                        metadata[key[5:]] = value

            tracers[name] = cls(name, spin, ell, beam,
                                quantity=quantity, beam_extra=beam_extra,
                                map_unit=map_unit, metadata=metadata)
        return tracers


class NuMapTracer(BaseTracer, tracer_type='NuMap'):
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

    @classmethod
    def to_tables(cls, instance_list):
        tables = []
        for tracer in instance_list:
            # Bandpasses
            names = ['nu', 'bandpass']
            cols = [tracer.nu, tracer.bandpass]
            for bandpass_id, col in tracer.bandpass_extra.items():
                names.append(str(bandpass_id))
                cols.append(col)
            table = Table(data=cols, names=names)
            table.meta['SACCTYPE'] = 'tracer'
            table.meta['SACCCLSS'] = cls.tracer_type
            table.meta['SACCNAME'] = tracer.name
            table.meta['SACCQTTY'] = tracer.quantity
            extname = f'tracer:{cls.tracer_type}:{tracer.name}:bandpass'
            table.meta['EXTNAME'] = extname
            table.meta['NU_UNIT'] = tracer.nu_unit
            table.meta['SPIN'] = tracer.spin
            for key, value in tracer.metadata.items():
                table.meta['META_'+key] = value
            remove_dict_null_values(table.meta)
            tables.append(table)

            # Beams
            names = ['ell', 'beam']
            cols = [tracer.ell, tracer.beam]
            for beam_id, col in tracer.beam_extra.items():
                names.append(str(beam_id))
                cols.append(col)
            table = Table(data=cols, names=names)
            table.meta['SACCTYPE'] = 'tracer'
            table.meta['SACCCLSS'] = cls.tracer_type
            table.meta['SACCNAME'] = tracer.name
            table.meta['SACCQTTY'] = tracer.quantity
            extname = f'tracer:{cls.tracer_type}:{tracer.name}:beam'
            table.meta['EXTNAME'] = extname
            table.meta['MAP_UNIT'] = tracer.map_unit
            table.meta['SPIN'] = tracer.spin
            for key, value in tracer.metadata.items():
                table.meta['META_'+key] = value
            remove_dict_null_values(table.meta)
            tables.append(table)
        return tables

    @classmethod
    def from_tables(cls, table_list):
        tracers = {}

        # Collect beam and bandpass tables describing the same tracer
        tr_tables = {}
        for table in table_list:
            # Read name and table type
            name = table.meta['SACCNAME']
            tabtyp = table.meta['EXTNAME'].split(':')[-1]
            if tabtyp not in ['bandpass', 'beam']:
                raise KeyError("Unknown table type " + table.meta['EXTNAME'])

            # If not present yet, create new tracer entry
            if name not in tr_tables:
                tr_tables[name] = {}
            # Add table
            tr_tables[name][tabtyp] = table

        # Now loop through different tracers and build them from their tables
        for dt in tr_tables.values():
            quantity = []
            metadata = {}
            nu = []
            bandpass = []
            bandpass_extra = {}
            nu_unit = 'GHz'
            map_unit = 'none'
            ell = []
            beam = []
            beam_extra = {}
            spin = 0

            if 'bandpass' in dt:
                table = dt['bandpass']
                name = table.meta['SACCNAME']
                quantity = table.meta.get('SACCQTTY', 'generic')
                nu = table['nu']
                bandpass = table['bandpass']
                for col in table.columns.values():
                    if col.name not in ['nu', 'bandpass']:
                        bandpass_extra[col.name] = col.data
                nu_unit = table.meta['NU_UNIT']
                spin = table.meta['SPIN']
                for key, value in table.meta.items():
                    if key.startswith("META_"):
                        metadata[key[5:]] = value

            if 'beam' in dt:
                table = dt['beam']
                name = table.meta['SACCNAME']
                quantity = table.meta.get('SACCQTTY', 'generic')
                ell = table['ell']
                beam = table['beam']
                for col in table.columns.values():
                    if col.name not in ['ell', 'beam']:
                        beam_extra[col.name] = col.data
                map_unit = table.meta['MAP_UNIT']
                spin = table.meta['SPIN']
                for key, value in table.meta.items():
                    if key.startswith("META_"):
                        metadata[key[5:]] = value

            tracers[name] = cls(name, spin,
                                nu, bandpass,
                                ell, beam,
                                quantity=quantity,
                                bandpass_extra=bandpass_extra,
                                beam_extra=beam_extra,
                                map_unit=map_unit,
                                nu_unit=nu_unit,
                                metadata=metadata)
        return tracers
