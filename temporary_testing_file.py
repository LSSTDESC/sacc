import sacc
from astropy.io import fits
from astropy.table import Table
import pprint
import sys


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

def load_fits(filename):
    with fits.open(filename) as f:
        tables = []
        for hdu in f:
            if 'SACCTYPE' in hdu.header and hdu.name.lower() != 'covariance':
                tables.append(Table.read(hdu))
    objs =  sacc.BaseIO.from_tables(tables)

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
    print(s.get_data_types())



for filename in sys.argv[1:]:
    print(f"Processing {filename}")
    go(filename)
    print("Done")
    print("")

