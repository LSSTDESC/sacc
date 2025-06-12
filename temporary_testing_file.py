import sacc
from sacc.utils import unique_list
from astropy.io import fits
from astropy.table import Table
import pprint
import numpy as np
import sys
from io import BytesIO

def fix_data_ordering(data_points):
    # Check if all tables have the 'sacc_ordering' column
    if not all('sacc_ordering' in dp.tags for dp in data_points):
        print("Warning: The FITS format without the 'sacc_ordering' column is deprecated")
        print("Assuming data rows are in the correct order as it was before version 1.0.")
        i = 0
        for dp in data_points:
            if 'sacc_ordering' in dp.tags:
                raise ValueError(
                    "Some data points have sacc ordering and some do not. Hybrid old/new version."
                    " This is very wrong. Please check your data files or ask on #desc-sacc for help."
                )

            dp.tags['sacc_ordering'] = i
            i += 1


    # Put the data back in its original order, matching the
    # covariance.
    ordered_data_points = [None for i in range(len(data_points))]
    for dp in data_points:
        i = dp.tags['sacc_ordering']
        ordered_data_points[i] = dp
        del dp.tags['sacc_ordering']

    return ordered_data_points

def save_fits(self, filename):


    objects = {
        "tracer": self.tracers,
        "data": self.data,
        "window": self._make_window_tables(),
     }
    tables = sacc.io.to_tables(objects)
    for table in tables:
        typ = table.meta['SACCTYPE']
        name = table.meta['SACCNAME']
        if typ == 'data':
            extname = f'{typ}:{name}'
        else:
            cls = table.meta['SACCCLSS']
            extname = f'{typ}:{cls}:{name}'
            table.meta['EXTNAME'] = extname


    # Create the actual fits object
    primary_header = fits.Header()

    # save any global metadata in the header.
    # We save the keys and values as separate header cards,
    # because otherwise the keys are all forced to upper case
    primary_header['NMETA'] = len(self.metadata)
    for i, (k, v) in enumerate(self.metadata.items()):
        primary_header[f'KEY{i}'] = k
        primary_header[f'VAL{i}'] = v
    hdus = [fits.PrimaryHDU(header=primary_header)] + \
            [fits.table_to_hdu(table) for table in tables]
    hdu_list = fits.HDUList(hdus)

    # Actuall write out data
    buf = BytesIO()
    hdu_list.writeto(buf)
    # Rewind and read the binary data we just wrote
    buf.seek(0)
    output_data = buf.read()
    # Write the binary data to the target file
    with open(filename, "wb") as f:
        f.write(output_data)


def load_fits(filename):
    with fits.open(filename) as f:
        tables = []
        for hdu in f:
            if hdu.name.lower() not in ['covariance', 'primary']:
                tables.append(Table.read(hdu))
    objs =  sacc.io.from_tables(tables)

    tracers = objs['tracer']
    data = objs['data']

    fix_data_ordering(data)

    s = sacc.Sacc()
    for tracer in tracers.values():
        s.add_tracer_object(tracer)

    for d in data:
        s.data.append(d)
    
    return s





def go(filename):
    if "raw" in filename:
        print("Skipping raw file:", filename)
        return
    s = load_fits(filename)
    save_fits(s, "new.fits")
    compare(filename, "new.fits")


def compare(filename, filename2):
    s = load_fits(filename)
    s2 = load_fits(filename2)

    # check they are equal
    assert np.allclose(s.get_mean(), s2.get_mean()), "Means do not match"
    assert s.tracers.keys() == s2.tracers.keys(), "Tracers do not match"
    i1 = s.get_tag('i')
    i2 = s2.get_tag('i')
    assert np.all(i1 == i2), "Tags do not match"

    for i in range(len(s2)):
        dp1 = s.data[i]
        dp2 = s2.data[i]
        assert dp1.tracers == dp2.tracers, f"Data point {i} tracers do not match"
        assert dp1.tags.keys() == dp2.tags.keys(), f"Data point {i} tags do not match: {dp1.tags} != {dp2.tags}"
        w1 = dp1.get_tag('window')
        w2 = dp2.get_tag('window')
        if isinstance(w1, sacc.BandpowerWindow):
            assert w1.nv == w2.nv, f"Data point {i} windows nv do not match: {w1.nv} != {w2.nv}"
            assert np.allclose(w1.values, w2.values), f"Data point {i} windows values do not match: {w1.values} != {w2.values}"
            assert np.allclose(w1.weight, w2.weight), f"Data point {i} windows weight do not match: {w1.weight} != {w2.weight}"
            assert w1.nell == w2.nell, f"Data point {i} windows nell do not match: {w1.nell} != {w2.nell}"

for filename in sys.argv[1:]:
    print(f"Processing {filename}")
    go(filename)

