import tempfile
import sacc
import sacc.data_types
import numpy as np
import pytest
import os
import pathlib
import urllib
import time

test_dir = pathlib.Path(__file__).resolve().parent
test_data_dir = test_dir / 'data'


# idea based on TreeCorr tests
def get_from_wiki(url):
    file_name = url.split('/')[-1]
    local_file_name = test_data_dir / file_name
    if not local_file_name.exists():
        print(f"Downlading {url} to data dir")
        try:
            urllib.request.urlretrieve(url, local_file_name)
        except urllib.request.HTTPError as err:
            if err.code == 429:
                print("Rate limit download - waiting 10 seconds to try again")
                time.sleep(10)
                urllib.request.urlretrieve(url, local_file_name)
            else:
                raise
    return local_file_name


def test_quantity_warning():
    s = sacc.Sacc()
    with pytest.warns(UserWarning):
        s.add_tracer('Misc', 'source_0',
                     quantity='dummy')


def test_data_type_warning():
    s = sacc.Sacc()
    s.add_tracer('Misc', 'source_0')
    with pytest.warns(UserWarning):
        s.add_data_point('cl_wrong', ('source_0', 'source_0'),
                         0.1, ell=10.)


def test_construct():
    s = sacc.Sacc()

    # Tracer
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2
    s.add_tracer('NZ', 'source_0', z, nz,
                 quantity='galaxy_density')

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0', 'source_0')
        s.add_data_point(sacc.standard_types.cl_00, tracers, ee, ell=10.0*i)


def test_tracers_later():
    s = sacc.Sacc()

    with pytest.raises(ValueError):
        tracers = ('source_0', 'source_0')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee,
                         tracers, 0.0, ell=1)

    s = sacc.Sacc()

    tracers = ('source_0', 'source_0')
    s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee,
                     tracers, 0.0, tracers_later=True, ell=1)


def test_full_cov():
    covmat = np.random.uniform(size=(100, 100))
    C = sacc.covariance.BaseCovariance.make(covmat)
    assert C.size == 100
    assert isinstance(C, sacc.covariance.FullCovariance)
    assert np.all(C.covmat == covmat)
    hdu = C.to_hdu()
    C2 = sacc.covariance.BaseCovariance.from_hdu(hdu)
    assert np.allclose(C.covmat, C2.covmat)


def test_block_cov():
    covmat = [np.random.uniform(size=(50, 50)),
              np.random.uniform(size=(100, 100)),
              np.random.uniform(size=(150, 150))]
    C = sacc.covariance.BaseCovariance.make(covmat)
    assert C.size == 300
    assert isinstance(C, sacc.covariance.BlockDiagonalCovariance)
    hdu = C.to_hdu()
    C2 = sacc.covariance.BaseCovariance.from_hdu(hdu)
    assert len(C2.blocks) == 3
    assert C.block_sizes == C2.block_sizes
    for i in range(3):
        assert np.allclose(C.blocks[i], C2.blocks[i])


def test_misc_tracer():
    md1 = {'potato': 'if_necessary', 'answer': 42, 'height': 1.83}
    md2 = {'potato': 'never'}
    T1 = sacc.BaseTracer.make('Misc', 'tracer1',
                              quantity='generic', metadata=md1)
    T2 = sacc.BaseTracer.make('Misc', 'tracer2',
                              quantity='generic', metadata=md2)
    assert T1.metadata == md1
    assert T2.metadata == md2

    tables = sacc.BaseTracer.to_tables([T1, T2])
    D = sacc.BaseTracer.from_tables(tables)

    T1a = D['tracer1']
    T2a = D['tracer2']
    assert T1a.metadata == md1
    assert T2a.metadata == md2


def test_numap_tracer():
    md1 = {'mac': 'yes', 'cheese': 'of_course', 'quantity': 3}
    md2 = {'mac': 'no', 'cheese': 'no'}
    ell = np.linspace(2, 1000, 1000)
    beam = np.exp(-0.1 * ell * (ell + 1))
    beam_extra = {'err1': np.sin(ell * 0.1)}
    nu = np.linspace(30., 60., 100)
    bandpass = np.ones(100)
    bandpass_extra = {'err1': bandpass * 0.1,
                      'err2': bandpass * 0.05}

    T1 = sacc.BaseTracer.make('NuMap', 'band1', 0,
                              nu, bandpass, ell, beam,
                              quantity='cmb_temperature',
                              bandpass_extra=bandpass_extra,
                              beam_extra=beam_extra,
                              metadata=md2)
    T2 = sacc.BaseTracer.make('NuMap', 'band2', 0,
                              nu, bandpass, ell, beam,
                              quantity='cmb_convergence',
                              bandpass_extra=bandpass_extra,
                              beam_extra=beam_extra,
                              metadata=md1)

    assert T2.metadata == md1
    assert T1.metadata == md2

    tables = sacc.BaseTracer.to_tables([T1, T2])
    D = sacc.BaseTracer.from_tables(tables)

    T1a = D['band1']
    T2a = D['band2']
    assert T1a.metadata == md2
    assert T2a.metadata == md1
    assert np.all(T1a.bandpass_extra['err1'] == 0.1 * bandpass)


def test_map_tracer():
    md1 = {'mac': 'yes', 'cheese': 'of_course', 'quantity': 3}
    md2 = {'mac': 'no', 'cheese': 'no'}
    ell = np.linspace(2, 1000, 1000)
    beam = np.exp(-0.1 * ell * (ell + 1))
    err = np.sin(ell * 0.1)
    beam_extra = {'err1': err}

    T1 = sacc.BaseTracer.make('Map', 'y_milca',
                              0, ell, beam,
                              quantity='cmb_tSZ',
                              beam_extra=beam_extra,
                              metadata=md1)
    T2 = sacc.BaseTracer.make('Map', 'y_nilc',
                              0, ell, beam,
                              quantity='cmb_kSZ',
                              beam_extra=beam_extra,
                              metadata=md2)

    assert T1.metadata == md1
    assert T2.metadata == md2

    tables = sacc.BaseTracer.to_tables([T1, T2])
    D = sacc.BaseTracer.from_tables(tables)

    T1a = D['y_milca']
    T2a = D['y_nilc']
    assert T1a.metadata == md1
    assert T2a.metadata == md2
    assert np.all(T1a.beam_extra['err1'] == err)


def test_nz_tracer():
    md1 = {'potato': 'if_necessary', 'answer': 42, 'height': 1.83}
    md2 = {'potato': 'never'}
    z = np.arange(0., 1., 0.01)
    Nz1 = 1*z  # not a sensible N(z)!
    Nz2 = 2*z  # not a sensible N(z)!

    Nz3 = 3*z
    Nz4 = 4*z

    more_nz = {'v1': Nz3, 'v2': Nz4}

    T1 = sacc.BaseTracer.make('NZ', 'tracer1', z, Nz1,
                              quantity='galaxy_density',
                              extra_columns=more_nz,
                              spin=0,
                              metadata=md1)
    T2 = sacc.BaseTracer.make('NZ', 'tracer2', z, Nz2,
                              quantity='galaxy_shear',
                              spin=2,
                              metadata=md2)
    assert T1.metadata == md1
    assert T2.metadata == md2

    tables = sacc.BaseTracer.to_tables([T1, T2])
    D = sacc.BaseTracer.from_tables(tables)

    T1a = D['tracer1']
    T2a = D['tracer2']
    assert T1a.metadata == md1
    assert T2a.metadata == md2

    assert np.all(T1a.extra_columns['v1'] == Nz3)
    assert np.all(T1a.extra_columns['v2'] == Nz4)


def test_mixed_tracers():
    md1 = {'potato': 'never'}
    md2 = {'rank': 'duke'}
    md3 = {'rank': 'earl', 'robes': 78}
    z = np.arange(0., 1., 0.01)
    Nz1 = 1*z  # not a sensible N(z)!
    Nz2 = 2*z
    T1 = sacc.BaseTracer.make('NZ', 'tracer1', z, Nz1,
                              quantity='galaxy_convergence')
    T2 = sacc.BaseTracer.make('NZ', 'tracer2', z, Nz2,
                              quantity='galaxy_shear', metadata=md1)

    M1 = sacc.BaseTracer.make("Misc", "sample1", metadata=md2)
    M2 = sacc.BaseTracer.make("Misc", "sample2", metadata=md3)

    tables = sacc.BaseTracer.to_tables([T1, M1, T2, M2])
    recovered = sacc.BaseTracer.from_tables(tables)
    assert recovered['sample1'].metadata['rank'] == 'duke'
    assert recovered['sample2'].metadata['robes'] == 78
    assert np.all(recovered['tracer1'].nz == Nz1)
    assert recovered['tracer2'].metadata['potato'] == 'never'


def test_inverses():
    N = 25
    C = np.random.uniform(0, 1, size=(N, N))
    C = (C+C.T) + np.eye(N)*20
    M1 = sacc.BaseCovariance.make(C)
    assert M1.size == N
    invC = M1.inverse
    ii = np.dot(invC, C)
    assert np.allclose(ii, np.eye(N))

    blocks = [np.random.uniform(0, 1, size=(5, 5))
              for i in range(5)]
    for b in blocks:
        b += b.T + np.eye(5)*20

    M2 = sacc.BaseCovariance.make(blocks)
    assert M2.size == N
    M2dense = np.zeros((N, N))
    for i in range(5):
        M2dense[i*5:i*5+5, i*5:i*5+5] = blocks[i]
    invC2 = M2.inverse
    ii = np.dot(invC2, M2dense)
    assert np.allclose(ii, np.eye(N))

    d = abs(np.random.uniform(0, 1, size=N))+1
    M3 = sacc.BaseCovariance.make(d)
    assert M3.size == N
    invC3 = M3.inverse
    assert np.count_nonzero(invC3 - np.diag(np.diagonal(invC3))) == 0
    assert np.allclose(invC3.diagonal() * d, 1)


def test_data_point():
    from sacc.data_types import DataPoint
    dt = sacc.data_types.standard_types.galaxy_shearDensity_cl_e
    value = 13.4
    tracers = ('aaa', 'bbb')
    tags = {'ell': 12, 'theta': 14.3}
    d = DataPoint(dt, tracers, value, **tags)
    s = repr(d)
    d2 = eval(s)
    assert d.tracers == d2.tracers
    assert d.tags == d2.tags
    assert d.data_type == d2.data_type
    assert d.value == d2.value


def test_keep_remove():
    s = sacc.Sacc()

    # Tracer
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2
    s.add_tracer('NZ', 'source_0', z, nz)
    s.add_tracer('NZ', 'source_1', z, nz,
                 quantity='galaxy_shear', spin=2)
    s.add_tracer('NZ', 'source_2', z, nz,
                 quantity='cluster_density')

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0', 'source_0')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee,
                         tracers, ee, ell=10.0*i)
    for i in range(20):
        bb = 0.1 * i
        tracers = ('source_1', 'source_1')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_bb,
                         tracers, bb, ell=10.0*i)
    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_2', 'source_2')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee,
                         tracers, ee, ell=10.0*i)

    # Select by data type
    s2 = s.copy()
    s2.keep_selection(data_type=sacc.standard_types.galaxy_shear_cl_bb)
    assert all(d.data_type == sacc.standard_types.galaxy_shear_cl_bb
               for d in s2.data)
    assert len(s2) == 20

    # From multiple tracers
    s2 = s.copy()
    s2.keep_selection(data_type=sacc.standard_types.galaxy_shear_cl_ee)
    assert all(d.data_type == sacc.standard_types.galaxy_shear_cl_ee
               for d in s2.data)
    assert len(s2) == 40

    # Test removing a single tracer
    s2 = s.copy()
    s2.remove_selection(tracers=('source_1', 'source_1'))
    for i, d in enumerate(s2.data):
        if i < 20:
            assert d.tracers == ('source_0', 'source_0')
        else:
            assert d.tracers == ('source_2', 'source_2')
    assert all(d.data_type == sacc.standard_types.galaxy_shear_cl_ee
               for d in s2.data)
    assert len(s2) == 40

    # Test selecting by tag
    s2 = s.copy()
    s2.remove_selection(ell__lt=55)
    ell = s2.get_tag('ell')
    for e in ell:
        assert e > 55
    s2 = s.copy()
    s2.keep_selection(ell__lt=55)
    ell = s2.get_tag('ell')
    for e in ell:
        assert e < 55

    # Cutting just by index
    s2 = s.copy()
    ind = s2.indices(tracers=('source_1', 'source_1'))
    assert (ind == np.arange(20, 40)).all()

    # multiple selections
    s2 = s.copy()
    ind = s2.indices(tracers=('source_2', 'source_2'), ell__lt=45)
    assert len(ind) == 5


def test_cutting_block_cov():
    covmat = [np.random.uniform(size=(50, 50)),
              np.random.uniform(size=(100, 100)),
              np.random.uniform(size=(150, 150))]
    C = sacc.covariance.BaseCovariance.make(covmat)
    ind = list(range(50))
    C2 = C.keeping_indices(np.arange(50))
    assert C2.size == len(ind)
    assert np.allclose(C2.get_block(ind), covmat[0])

def test_cutting_block_cov2():
    covmat = [np.random.uniform(size=(50, 50)),
              np.random.uniform(size=(100, 100)),
              np.random.uniform(size=(150, 150))]
    C = sacc.covariance.BaseCovariance.make(covmat)
    ind = list(range(50,150))
    C2 = C.keeping_indices(np.arange(50,150))
    assert C2.size == len(ind)
    assert np.allclose(C2.get_block(range(100)), covmat[1])


def test_cutting_full_cov():
    covmat = np.random.uniform(size=(100, 100))
    C = sacc.covariance.BaseCovariance.make(covmat)
    ind = np.arange(10, dtype=int)
    C2 = C.keeping_indices(ind)
    assert np.allclose(C2.get_block(ind),
                       covmat[:10, :10])


def test_cutting_diag_cov():
    diag = np.random.uniform(size=(100,))
    C = sacc.covariance.BaseCovariance.make(diag)
    ind = np.arange(20, dtype=int)
    C2 = C.keeping_indices(ind)
    assert np.allclose(C2.get_block(ind).diagonal(), diag[:20])


def test_parse_data_names():
    for name in sacc.data_types.required_tags_verbose:
        sources, props, stat, sub = sacc.parse_data_type_name(name)
        name2 = sacc.build_data_type_name(sources, props, stat, sub)
        assert name == name2


def test_bandpower_window():
    nb = 20
    nl = 200
    dl = nl // nb
    ells = np.arange(nl)
    w = np.zeros([nb, nl])
    for i in range(nb):
        w[i, i*dl: (i+1)*dl] = 1./dl

    W1 = [sacc.BandpowerWindow(ells, w.T)]

    tables = sacc.BandpowerWindow.to_tables(W1)
    W2 = sacc.BandpowerWindow.from_tables(tables)
    for w1 in W1:
        w2 = W2[id(w1)]
        assert np.all(w1.values == w2.values)
        assert np.all(w1.weight.flatten() == w2.weight.flatten())


def test_tophat_window():
    edges = np.arange(10) * 10
    W1 = [sacc.TopHatWindow(edges[:-1], edges[1:])]

    tables = sacc.TopHatWindow.to_tables(W1)
    W2 = sacc.TopHatWindow.from_tables(tables)
    for w1 in W1:
        w2 = W2[id(w1)]
        assert np.all(w1.min == w2.min)
        assert np.all(w1.max == w2.max)


def test_log_window():
    edges = (np.arange(10) + 1) * 10
    W1 = [sacc.LogTopHatWindow(edges[:-1], edges[1:])]

    tables = sacc.LogTopHatWindow.to_tables(W1)
    W2 = sacc.LogTopHatWindow.from_tables(tables)
    for w1 in W1:
        w2 = W2[id(w1)]
        assert np.all(w1.min == w2.min)
        assert np.all(w1.max == w2.max)


def test_concatenate_covariance():
    v1 = np.array([1., 2., 3.])
    v2 = np.array([4.])
    A = sacc.BaseCovariance.make(v1)
    B = sacc.BaseCovariance.make(v2)
    C = sacc.covariance.concatenate_covariances(A, B)
    assert isinstance(C, sacc.covariance.DiagonalCovariance)
    assert np.allclose(C.diag, [1, 2, 3, 4])

    v1 = np.array([2.])
    v2 = np.array([[3., 0.1], [0.1, 3]])

    A = sacc.BaseCovariance.make(v1)
    B = sacc.BaseCovariance.make(v2)
    C = sacc.covariance.concatenate_covariances(A, B)
    assert isinstance(C, sacc.covariance.BlockDiagonalCovariance)
    test_C = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.1],
        [0.0, 0.1, 3.0]]
        )
    assert np.allclose(C.dense, test_C)

    v1 = np.array([[2.0, 0.2, ],
                   [0.2, 3.0, ]])
    v2 = np.array([[4.0, -0.2, ],
                   [-0.2, 5.0, ]])
    test_C = np.array([[2.0, 0.2, 0.0, 0.0],
                       [0.2, 3.0, 0.0, 0.0],
                       [0.0, 0.0, 4.0, -0.2],
                       [0.0, 0.0, -0.2, 5.0]])

    A = sacc.BaseCovariance.make(v1)
    B = sacc.BaseCovariance.make(v2)
    C = sacc.covariance.concatenate_covariances(A, B)
    assert isinstance(C, sacc.covariance.BlockDiagonalCovariance)
    assert np.allclose(C.dense, test_C)


def test_concatenate_data():
    s1 = sacc.Sacc()

    # Tracer
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2
    s1.add_tracer('NZ', 'source_0', z, nz)

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0', 'source_0')
        s1.add_data_point(sacc.standard_types.galaxy_shear_cl_ee,
                          tracers, ee, ell=10.0*i)

    s2 = sacc.Sacc()

    # Tracer
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2
    s2.add_tracer('NZ', 'source_0', z, nz,
                  quantity='galaxy_shear', spin=2)

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0', 'source_0')
        s2.add_data_point(sacc.standard_types.galaxy_shear_cl_ee,
                          tracers, ee, ell=10.0*i, label='xxx')

    # name clash
    with pytest.raises(ValueError):
        sacc.concatenate_data_sets(s1, s2)

    s3 = sacc.concatenate_data_sets(s1, s2, labels=['1', '2'])
    assert 'source_0_1' in s3.tracers
    assert 'source_0_2' in s3.tracers
    assert len(s3) == len(s1) + len(s2)

    # check data points in right order
    for i in range(20):
        assert s3.data[i].get_tag('ell') == 10.0*i
        assert s3.data[i+20].get_tag('ell') == 10.0*i
        assert s3.data[i].get_tag('label') == '1'
        assert s3.data[i+20].get_tag('label') == 'xxx_2'
        t1 = s3.data[i].tracers[0]
        t2 = s3.data[i+20].tracers[0]
        assert t1 == 'source_0_1'
        assert t2 == 'source_0_2'
        s3.get_tracer(t1)
        s3.get_tracer(t2)


def test_io():
    s = sacc.Sacc()

    # Tracer
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2
    s.add_tracer('NZ', 'source_0', z, nz)

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0', 'source_0')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee,
                         tracers, ee, ell=10.0*i)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test.sacc')
        s.save_fits(filename)
        s2 = sacc.Sacc.load_fits(filename)

    assert len(s2) == 20
    mu = s2.get_mean(sacc.standard_types.galaxy_shear_cl_ee)
    for i in range(20):
        assert mu[i] == 0.1 * i


def test_io_maps_bpws():
    s = sacc.Sacc()

    n_ell = 10
    d_ell = 100
    n_ell_large = n_ell * d_ell
    ell = np.linspace(2, 1000, n_ell)
    c_ell = 1./(ell+1)**3
    beam = np.exp(-0.1 * ell * (ell+1))
    nu = np.linspace(30., 60., 100)
    bandpass = np.ones(100)
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2

    # Tracer
    s.add_tracer('NZ', 'gc', z, nz)
    s.add_tracer('NuMap', 'cmbp', 2, nu, bandpass, ell, beam)
    s.add_tracer('Map', 'sz', 0, ell, beam)

    # Window
    ells_large = np.arange(n_ell_large)
    window_single = np.zeros([n_ell, n_ell_large])
    for i in range(n_ell):
        window_single[i, i * d_ell: (i + 1) * d_ell] = 1.
    wins = sacc.BandpowerWindow(ells_large, window_single.T)

    s.add_ell_cl('cl_00', 'gc', 'gc', ell, c_ell, window=wins)
    s.add_ell_cl('cl_0e', 'gc', 'cmbp', ell, c_ell, window=wins)
    s.add_ell_cl('cl_00', 'gc', 'sz', ell, c_ell, window=wins)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test.sacc')
        s.save_fits(filename)
        s2 = sacc.Sacc.load_fits(filename)

    assert len(s2) == 30
    l, cl, ind = s2.get_ell_cl('cl_00', 'gc', 'sz',
                               return_ind=True)
    w = s2.get_bandpower_windows(ind)
    assert np.all(cl == c_ell)
    assert w.weight.shape == (n_ell_large, n_ell)


@pytest.mark.parametrize("vv,ncl,ntr",
                         [('0.2.0', 2, 2),
                          ('0.3.0', 3, 2),
                          ('0.4.2', 6, 5)])
def test_legacy_format(vv, ncl, ntr):
    print(vv, ncl, ntr)
    local_file_name = get_from_wiki(
        f'https://github.com/LSSTDESC/sacc/wiki/legacy_files/dummy_v{vv}.fits')
    s = sacc.Sacc.load_fits(local_file_name)
    assert len(s.mean) == ncl * 100
    assert len(s.tracers) == ntr
