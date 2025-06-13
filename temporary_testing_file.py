import sacc
import numpy as np
import sys




def go(filename):
    if "raw" in filename:
        print("Skipping raw file:", filename)
        print("")
        return
    print("")
    print("Loading original file:", filename)
    s = sacc.Sacc.load_fits(filename)
    print("")
    print("Original file loaded successfully.")
    print("Saving to new file: new.fits")
    sacc.Sacc.save_fits(s, "new.fits", overwrite=True)
    compare(filename, "new.fits")


def compare(filename, filename2):
    print("")
    print("Re-loading ", filename)
    s = sacc.Sacc.load_fits(filename)
    print("")
    print("Loading ", filename2)
    s2 = sacc.Sacc.load_fits(filename2)

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

    if s.covariance is None:
        assert s2.covariance is None, "Covariance is None in one file but not the other"
    else:
        assert np.allclose(s.covariance.dense, s2.covariance.dense), "Covariance matrices do not match"

for filename in sys.argv[1:]:
    print(f"Processing {filename}")
    go(filename)

