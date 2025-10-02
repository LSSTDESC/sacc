import tempfile
import sacc
import sacc.data_types
import sacc.tracers
import numpy as np
import pytest
import os
import pathlib
import urllib
import time
import warnings
try:
    import qp
    QP_AVAILABLE = True
except:
    QP_AVAILABLE = False

# Decorator to skip tests if qp is not available.
skip_if_no_qp = pytest.mark.skipif(not QP_AVAILABLE, reason="qp not available")

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


@pytest.fixture
def filled_sacc():
    s = sacc.Sacc()

    # Tracer
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2
    s.add_tracer('NZ', 'source_0', z, nz)
    s.add_tracer('NZ', 'source_1', z, nz,
                 quantity='cluster_density')
    s.add_tracer('NZ', 'source_2', z, nz,
                 quantity='cluster_density')

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0',)
        s.add_data_point(sacc.standard_types.count,
                         tracers, ee, ell=10.0*i)
    for i in range(20):
        bb = 0.2 * i
        tracers = ('source_0', 'source_1')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_bb,
                         tracers, bb, ell=10.0*i)
    for i in range(20):
        ee = 0.3 * i
        tracers = ('source_1', 'source_1', 'source_0')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee,
                         tracers, ee, ell=10.0*i)
    for i in range(20):
        bb = 0.4 * i
        tracers = ('source_2', 'source_2')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_bb,
                         tracers, bb, ell=10.0*i)

    return s

def make_full_cov(n):
    """
    Make a full covariance matrix of shape (n, n)
    """
    cov = np.random.uniform(size=(n, n))
    cov = (cov + cov.T) / 2  # make symmetric
    return cov

def test_add_covariance(filled_sacc):
    s = filled_sacc.copy()
    cov = np.ones((s.mean.size, s.mean.size))
    s.add_covariance(cov)

    with pytest.raises(RuntimeError):
        s.add_covariance(cov)

    s.add_covariance(0 * cov, overwrite=True)
    assert np.all(s.covariance.covmat == 0 * cov)

def test_get_sigma(filled_sacc):
    n = len(filled_sacc)

    # first check it works with a diagonal cov
    # for the full matrix
    s = filled_sacc.copy()
    variance = np.random.uniform(size=n)
    s.add_covariance(variance)
    assert isinstance(s.covariance, sacc.covariance.DiagonalCovariance)
    sigma = s.get_standard_deviation()
    assert np.allclose(sigma, variance ** 0.5)

    # Now check the last 20 data points work
    tracers = ('source_2', 'source_2')
    dt = sacc.standard_types.galaxy_shear_cl_bb
    sigma = s.get_standard_deviation(tracers=tracers, data_type=dt)
    assert len(sigma) == 20
    assert np.allclose(sigma, variance[-20:] ** 0.5)

    # Now check with a dense covariance
    s = filled_sacc.copy()
    cov = make_full_cov(n)
    s.add_covariance(cov)
    assert isinstance(s.covariance, sacc.covariance.FullCovariance)

    sigma = s.get_standard_deviation()
    assert np.allclose(sigma, np.sqrt(np.diagonal(cov)))

    tracers = ('source_2', 'source_2')
    dt = sacc.standard_types.galaxy_shear_cl_bb
    sigma = s.get_standard_deviation(tracers=tracers, data_type=dt)
    assert len(sigma) == 20
    assert np.allclose(sigma, np.sqrt(np.diagonal(cov))[-20:])

    # now block-diagonal
    s = filled_sacc.copy()
    cov = [
        np.random.uniform(size=(n//4, n//4))
        for i in range(4)
    ]
    s.add_covariance(cov)
    assert isinstance(s.covariance, sacc.covariance.BlockDiagonalCovariance)
    sigma = s.get_standard_deviation()
    expected = np.concatenate([
        np.sqrt(np.diagonal(c))
        for c in cov
    ])
    assert np.allclose(sigma, expected)



def test_get_data_types(filled_sacc):
    s = filled_sacc
    dt1 = [sacc.standard_types.count, sacc.standard_types.galaxy_shear_cl_bb,
           sacc.standard_types.galaxy_shear_cl_ee]
    dt2 = s.get_data_types()
    assert sorted(dt1) == sorted(dt2)

    dt2 = s.get_data_types(tracers=('source_0', 'source_1'))
    assert [sacc.standard_types.galaxy_shear_cl_bb] == dt2



def test_table_creation_misc_1_instance():
    t = sacc.tracers.MiscTracer("source_0")
    tables = sacc.tracers.MiscTracer.to_table([t])
    assert len(tables) == 1


def test_table_creation_misc_2_instances():
    t1 = sacc.tracers.MiscTracer("source_0")
    t2 = sacc.tracers.MiscTracer("source_1")
    table = sacc.tracers.MiscTracer.to_table([t1, t2])
    assert len(table) == 2


def test_table_creation_nz_1_instance():
    t = sacc.tracers.NZTracer("source_0", np.zeros(1), np.zeros(1))
    tables = t.to_table()
    assert len(tables) == 1


def test_table_creation_nz_2_instances():
    t1 = sacc.tracers.NZTracer("source_0", np.zeros(17), np.zeros(17))
    table = t1.to_table()
    assert len(table) == 17

def test_table_creation_map_1_instance():
    t = sacc.tracers.MapTracer("source_0", 1, [1.5], [10.0])
    table = t.to_table()
    assert np.allclose(table['ell'], [1.5])
    assert np.allclose(table['beam'], [10.0])
2


def test_table_creation_numap_1_instance():
    t = sacc.tracers.NuMapTracer("source_0", 1, [1.5], [10.0], np.zeros(1), np.zeros(1))
    tables = t.to_tables()
    assert len(tables) == 2



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
    table = C.to_table()
    C2 = sacc.covariance.FullCovariance.from_table(table)
    assert np.allclose(C.covmat, C2.covmat)


def test_block_cov():
    covmat = [np.random.uniform(size=(50, 50)),
              np.random.uniform(size=(100, 100)),
              np.random.uniform(size=(150, 150))]
    C = sacc.covariance.BaseCovariance.make(covmat)
    assert C.size == 300
    assert isinstance(C, sacc.covariance.BlockDiagonalCovariance)
    tables = C.to_tables()
    C2 = sacc.covariance.BlockDiagonalCovariance.from_tables(tables)
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

    table = sacc.tracers.MiscTracer.to_table([T1, T2])
    D = sacc.tracers.MiscTracer.from_table(table)

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

    tables1 = T1.to_tables()
    tables2 = T2.to_tables()

    T1a = sacc.tracers.NuMapTracer.from_tables(tables1)
    T2a = sacc.tracers.NuMapTracer.from_tables(tables2)
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

    T1a = sacc.tracers.MapTracer.from_table(T1.to_table())
    T2a = sacc.tracers.MapTracer.from_table(T2.to_table())
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

    # check that we can make a tracer using lower case
    T3 = sacc.BaseTracer.make('nz', 'tracer2', z, Nz2,
                              quantity='galaxy_shear',
                              spin=2,
                              metadata=md2)
    assert T3.metadata == md2


    T1a = sacc.tracers.NZTracer.from_table(T1.to_table())
    T2a = sacc.tracers.NZTracer.from_table(T2.to_table())
    assert T1a.metadata == md1
    assert T2a.metadata == md2

    assert np.all(T1a.extra_columns['v1'] == Nz3)
    assert np.all(T1a.extra_columns['v2'] == Nz4)



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
    s.add_tracer('nZ', 'source_1', z, nz,
                 quantity='galaxy_shear', spin=2)
    s.add_tracer('nz', 'source_2', z, nz,
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


def test_remove_keep_tracers(filled_sacc):
    s = filled_sacc.copy()

    s.remove_tracers(['source_0'])

    assert ['source_1', 'source_2'] == list(s.tracers.keys())
    assert [('source_2', 'source_2')] == s.get_tracer_combinations()

    s = filled_sacc.copy()
    s.keep_tracers(['source_0'])
    assert ['source_0'] == list(s.tracers.keys())
    assert [('source_0',)] == s.get_tracer_combinations()


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
    ind = list(range(50, 150))
    C2 = C.keeping_indices(np.arange(50, 150))
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

    w1 = sacc.BandpowerWindow(ells, w.T)

    table = w1.to_table()
    w2 = sacc.BandpowerWindow.from_table(table)
    assert np.all(w1.values == w2.values)
    assert np.all(w1.weight.flatten() == w2.weight.flatten())


def test_tophat_window():
    edges = np.arange(10) * 10
    W1 = [sacc.TopHatWindow(edges[:-1], edges[1:])]

    table = sacc.TopHatWindow.to_table(W1)
    W2 = sacc.TopHatWindow.from_table(table)
    for w1 in W1:
        w2 = W2[id(w1)]
        assert np.all(w1.min == w2.min)
        assert np.all(w1.max == w2.max)


def test_log_window():
    edges = (np.arange(10) + 1) * 10
    W1 = [sacc.LogTopHatWindow(edges[:-1], edges[1:])]

    tables = sacc.LogTopHatWindow.to_table(W1)
    W2 = sacc.LogTopHatWindow.from_table(tables)
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

    # same tracer
    s3 = sacc.concatenate_data_sets(s1, s2, same_tracers=['source_0'])
    assert ['source_0'] == list(s3.tracers.keys())

    # check data points in right order
    for i in range(20):
        assert s3.data[i].get_tag('ell') == 10.0*i
        assert s3.data[i+20].get_tag('ell') == 10.0*i
        assert s3.data[i].get_tag('label') is None
        assert s3.data[i+20].get_tag('label') == 'xxx'
        t1 = s3.data[i].tracers[0]
        t2 = s3.data[i+20].tracers[0]
        assert t1 == 'source_0'
        assert t1 == t2
        # To make sure the first 'source_0' tracer is used and not rewritten
        s3.get_tracer(t1).quantity == 'generic'

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

    # labels + same_tracers
    s4 = sacc.concatenate_data_sets(s3, s3, labels=['x', 'y'],
                                    same_tracers=['source_0_1'])
    trs = ['source_0_1', 'source_0_2_x', 'source_0_2_y']
    assert trs == list(s4.tracers.keys())
    assert s4.mean.size == 2 * s3.mean.size


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
        sacc_filename = os.path.join(tmpdir, 'test.sacc')
        hdf_filename = os.path.join(tmpdir, 'test.hdf5')
        s.save_fits(sacc_filename)
        s.save_hdf5(hdf_filename)
        s2 = sacc.Sacc.load_fits(sacc_filename)
        s3 = sacc.Sacc.load_hdf5(hdf_filename)
        assert s2 == s
        assert s3 == s

    for sn in [s2, s3]:
        mu = sn.get_mean(sacc.standard_types.galaxy_shear_cl_ee)
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
    s.add_tracer('maP', 'sz', 0, ell, beam)

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
        sacc_filename = os.path.join(tmpdir, 'test.sacc')
        hdf_filename = os.path.join(tmpdir, 'test.hdf5')
        s.save_fits(sacc_filename)
        s.save_hdf5(hdf_filename)
        s2 = sacc.Sacc.load_fits(sacc_filename)
        s3 = sacc.Sacc.load_hdf5(hdf_filename)
        assert s2 == s
        assert s3 == s

    assert len(s2) == 30
    l, cl, ind = s2.get_ell_cl('cl_00', 'gc', 'sz',
                               return_ind=True)
    w = s2.get_bandpower_windows(ind)
    assert np.all(cl == c_ell)
    assert w.weight.shape == (n_ell_large, n_ell)

def test_rename_tracer(filled_sacc):
    s = filled_sacc.copy()

    tracer_comb = s.get_tracer_combinations()

    s.rename_tracer('source_0', 'src_0')

    # Check that the name attribute of the tracer has changed
    tr = s.get_tracer('src_0')
    assert tr.name == 'src_0'

    # Check that the data tracer combinations have been updated
    tracer_comb_new = s.get_tracer_combinations()
    for trs, trs_new in zip(tracer_comb, tracer_comb_new):
        trs_n = []
        for tri in trs:
            if 'source_0' == tri:
                tri = 'src_0'
            trs_n.append(tri)

        assert tuple(trs_n) == trs_new

    # Check it is permanent
    with tempfile.TemporaryDirectory() as tmpdir:
        fits_filename = os.path.join(tmpdir, 'test.sacc')
        hdf_filename = os.path.join(tmpdir, 'test.hdf5')
        s.save_fits(fits_filename)
        s.save_hdf5(hdf_filename)
        s2 = sacc.Sacc.load_fits(fits_filename)
        s3 = sacc.Sacc.load_hdf5(hdf_filename)
        assert s2 == s
        assert s3 == s

    assert ('source_0' not in s2.tracers) and ('src_0' in s2.tracers)
    assert sorted(tracer_comb_new) == sorted(s2.get_tracer_combinations())
    # Make sure the tracers are ordered the same way when comparing the data
    # points
    s.to_canonical_order()
    s2.to_canonical_order()
    s3.to_canonical_order()
    assert s2 == s
    assert s3 == s
    assert np.all(s.mean == s2.mean)
    assert np.all(s.mean == s3.mean)

    assert np.all(s.indices(tracers=('src_0', 'source_1')) ==
              s2.indices(tracers=('src_0', 'source_1')))
    assert np.all(s.indices(tracers=('src_0', 'source_1')) ==
              s3.indices(tracers=('src_0', 'source_1')))

    assert np.all(s.indices(tracers=('source_1', 'source_1', 'src_0')) ==
                  s2.indices(tracers=('source_1', 'source_1', 'src_0')))
    assert np.all(s.indices(tracers=('source_1', 'source_1', 'src_0')) ==
                  s3.indices(tracers=('source_1', 'source_1', 'src_0')))


@skip_if_no_qp
def test_qpnz_tracer():
    md1 = {'potato': 'if_necessary', 'answer': 42, 'height': 1.83}
    md2 = {'potato': 'never'}
    z = np.linspace(0., 1., 101)

    nz_qp_interp = qp.Ensemble(qp.interp, data=dict(xvals=z, yvals=np.ones(shape=(1, 101))))
    nz_qp_hist = qp.Ensemble(qp.hist, data=dict(bins=z, pdfs=np.ones(shape=(1, 100))))

    T1 = sacc.BaseTracer.make('QPNZ', 'tracer1', nz_qp_interp, z,
                              quantity='galaxy_density',
                              metadata=md1)
    T2 = sacc.BaseTracer.make('QPNZ', 'tracer2', nz_qp_hist, z,
                              quantity='galaxy_shear',
                              metadata=md2)
    assert T1.metadata == md1
    assert T2.metadata == md2

    tables1 = T1.to_tables()
    tables2 = T2.to_tables()

    T1a = sacc.tracers.QPNZTracer.from_tables(tables1)
    T2a = sacc.tracers.QPNZTracer.from_tables(tables2)
    assert T1a.metadata == md1
    assert T2a.metadata == md2

    # test version without saved z
    T3 = sacc.BaseTracer.make('QPNZ', 'tracer3', nz_qp_interp,
                              quantity='galaxy_density',
                              metadata=md1)
    tables = T3.to_tables()
    D = sacc.tracers.QPNZTracer.from_tables(tables)
    assert D.z is None
    assert D.name == 'tracer3'


@skip_if_no_qp
def test_io_qp():
    s = sacc.Sacc()

    # Tracer
    z = np.linspace(0., 1.0, 101)
    nz = np.expand_dims((z-0.5)**2/0.1**2, 0)
    ens = qp.Ensemble(qp.interp, data=dict(xvals=z, yvals=nz))
    mu = ens.mean()
    ancil = {
        "means": np.array([mu]),
    }
    ens.set_ancil(ancil)
    s.add_tracer('QpnZ', 'source_0', ens, z)

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0', 'source_0')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee,
                         tracers, ee, ell=10.0*i)

    with tempfile.TemporaryDirectory() as tmpdir:
        fits_filename = os.path.join(tmpdir, 'test.sacc')
        hdf_filename = os.path.join(tmpdir, 'test.hdf5')
        s.save_fits(fits_filename)
        s.save_hdf5(hdf_filename)
        s2 = sacc.Sacc.load_fits(fits_filename)
        s3 = sacc.Sacc.load_hdf5(hdf_filename)
        assert s2 == s
        assert s3 == s

    assert len(s2) == 20
    mu = s2.get_mean(sacc.standard_types.galaxy_shear_cl_ee)
    for i in range(20):
        assert mu[i] == 0.1 * i

    assert len(s3) == 20
    mu = s3.get_mean(sacc.standard_types.galaxy_shear_cl_ee)
    for i in range(20):
        assert mu[i] == 0.1 * i

def test_sacc_has_tracer(filled_sacc):
    s = filled_sacc
    assert not s.has_tracer("this_is_not_a_tracer")
    for tracer_name in ['source_0', 'source_1', 'source_2']:
        assert s.has_tracer(tracer_name)

def test_save_order_maintained():
    s = sacc.Sacc()

    # Tracer
    s.add_tracer('misc', 'source_0')

    # add a series of data points with alternating
    # types so they will be saved in separate table
    s.add_data_point("dt1", ('source_0',), 0.1, a=1)
    s.add_data_point("dt2", ('source_0',), 0.1, b=2)
    s.add_data_point("dt1", ('source_0',), 0.1, a=-1)
    s.add_data_point("dt2", ('source_0',), 0.1, b=-2)

    with tempfile.TemporaryDirectory() as tmpdir:
        fits_filename = os.path.join(tmpdir, 'test.sacc')
        hdf_filename = os.path.join(tmpdir, 'test.hdf5')
        s.save_fits(fits_filename)
        s.save_hdf5(hdf_filename)
        s2 = sacc.Sacc.load_fits(fits_filename)
        s3 = sacc.Sacc.load_hdf5(hdf_filename)
        assert s2 == s
        assert s3 == s

    # check that the order of the data points is maintained
    for ss in [s2, s3]:
        assert len(ss) == 4
        assert ss.data[0].data_type == "dt1"
        assert ss.data[1].data_type == "dt2"
        assert ss.data[2].data_type == "dt1"
        assert ss.data[3].data_type == "dt2"
        # and that the tags are all okay
        assert ss.data[0].get_tag('a') == 1
        assert ss.data[1].get_tag('b') == 2
        assert ss.data[2].get_tag('a') == -1
        assert ss.data[3].get_tag('b') == -2

def test_warn_empty():
    s = sacc.Sacc()

    # First check that no warning is raised if warn_empty is False
    # which is the default
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s.indices(data_type='non_existent_data_type', warn_empty=False)

    # Now check that a warning is raised if warn_empty is True
    with pytest.warns(UserWarning, match="Empty index selected"):
        s.indices(data_type='non_existent_data_type', warn_empty=True)

@skip_if_no_qp
def test_old_qp_sacc_readable():
    s = sacc.Sacc.load_fits("test/legacy_files/old_qp_sacc.fits")
    t = s.tracers['tracer1']
    assert isinstance(t, sacc.tracers.QPNZTracer)
    assert t.quantity
    md1 = {'potato': 'if_necessary', 'answer': 42, 'height': 1.83}
    for k, v in md1.items():
        assert t.metadata[k] == v

def test_metadata_round_trip():
    s = sacc.Sacc()
    s.metadata["mouse"]  = True
    s.metadata["rat"] = False
    s.metadata["cats"]  = "good"
    s.metadata["dogs"]  = "bad"
    s.metadata["number"] = 42
    s.metadata["pi"] = 3.14159

    with tempfile.TemporaryDirectory() as tmpdir:
        fits_filename = os.path.join(tmpdir, 'test_metadata_round_trip.sacc')
        hdf_filename = os.path.join(tmpdir, 'test_metadata_round_trip.hdf5')
        s.save_fits(fits_filename)
        s.save_hdf5(hdf_filename)
        s2 = sacc.Sacc.load_fits(fits_filename)
        s3 = sacc.Sacc.load_hdf5(hdf_filename)
        assert s2 == s
        assert s3 == s

    for sn in [s2, s3]:
        assert sn.metadata["mouse"] is True
        assert sn.metadata["rat"] is False
        assert sn.metadata["cats"] == "good"
        assert sn.metadata["dogs"] == "bad"
        assert sn.metadata["number"] == 42
        assert sn.metadata["pi"] == 3.14159

def test_equality_empty():
    x = sacc.Sacc()
    assert x == x

    y = sacc.Sacc()
    assert x is not y
    assert x == y


def test_equality_sacc_with_covariance(filled_sacc):
    x = filled_sacc.copy()
    cov = make_full_cov(len(filled_sacc))
    x.add_covariance(cov.copy())
    assert x == x

    y = filled_sacc.copy()
    y.add_covariance(cov.copy())
    assert x is not y
    assert x == y

    y.remove_tracers(['source_0'])
    assert x != y

def test_equality_mismatched_tracers(filled_sacc):
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2
    x = filled_sacc.copy()

    # If add a tracer, x and y should not longer be equal
    y = filled_sacc.copy()
    assert x == y
    y.add_tracer('NZ', 'source_100', z, nz)
    assert x != y

    # If we modify a tracer, x and y should not longer be equal
    y = filled_sacc.copy()
    assert x == y
    nz = y.tracers['source_0'].nz.copy()
    nz += 1.0
    assert np.all(y.tracers['source_0'].nz != nz)

    y.tracers['source_0'].nz = nz
    assert np.all(y.tracers['source_0'].nz != x.tracers['source_0'].nz)
    assert x != y

def test_equality_mismatched_datapoints(filled_sacc):
    x = filled_sacc.copy()
    y = filled_sacc.copy()
    assert x == y

    # Adding a datapoint should also make them not equal
    y.add_data_point("dt1", ('source_100',), 0.5, tracers_later=True)
    assert x != y

    # Adding the same data point to x should make them equal again
    x.add_data_point("dt1", ('source_100',), 0.5, tracers_later=True)

    # Changing any data point value should also make them not equal
    data = y.get_data_points()
    last = data[-1]
    assert last.value == 0.5
    last.value *= 2
    assert last.value == 1.0
    assert x != y


def test_equality_wrong_type():
    x = sacc.Sacc()
    y = 0
    assert x != y

def test_equality_mismatched_tracer_names():
    x = sacc.Sacc()
    y = sacc.Sacc()
    x.add_tracer('NZ', 'source_0', np.array([0., 1.]), np.array([1., 1.]))
    y.add_tracer('NZ', 'source_1', np.array([0., 1.]), np.array([1., 1.]))
    assert x != y

def test_nzshift_saving():
    s = sacc.Sacc()
    z = np.linspace(0, 1, 100)
    nz1 = np.exp(-((z - 0.3) ** 2) / (0.1 ** 2))
    nz2 = np.exp(-((z - 0.6) ** 2) / (0.1 ** 2))
    s.add_tracer('NZ', 'source1', z, nz1)
    s.add_tracer('NZ', 'source2', z, nz2)

    tracer_names = ["source1", "source2"]
    mu = [0.001, 0.002]

    # check with a 1D array for cholesky, representing a diagonal covariance
    cholesky = [0.01, 0.02]
    shift = sacc.NZShiftUncertainty("shift", tracer_names, mu, cholesky)
    s.add_tracer_uncertainty_object(shift)


    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_nzshift.sacc')
        s.save_fits(filename)
        s2a = sacc.Sacc.load_fits(filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_nzshift.sacc')
        s.save_hdf5(filename)
        s2b = sacc.Sacc.load_hdf5(filename)

    for s2 in [s2a, s2b]:
        assert len(s2.tracer_uncertainties) == 1
        assert "shift" in s2.tracer_uncertainties
        shift2 = s2.tracer_uncertainties["shift"]
        assert shift2.name == "shift"
        assert isinstance(shift2, sacc.NZShiftUncertainty)
        assert shift2.tracer_names == tracer_names
        assert np.allclose(shift2.mean, mu)
        assert np.allclose(shift2.linear_transformation, np.diag(cholesky))


def test_nzshift_stretch_saving():
    s = sacc.Sacc()
    z = np.linspace(0, 1, 100)
    nz1 = np.exp(-((z - 0.3) ** 2) / (0.1 ** 2))
    nz2 = np.exp(-((z - 0.6) ** 2) / (0.1 ** 2))
    s.add_tracer('NZ', 'source1', z, nz1)
    s.add_tracer('NZ', 'source2', z, nz2)

    tracer_names = ["source1", "source2"]
    mu = [0.001, 0.002]

    # this time test with a matrix
    chol = np.array([[0.01, 0], [0, 0.02]])
    shift = sacc.NZShiftStretchUncertainty("shift", tracer_names, mu, chol)
    s.add_tracer_uncertainty_object(shift)


    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_nzshift.sacc')
        s.save_fits(filename)
        s2 = sacc.Sacc.load_fits(filename)

    assert len(s2.tracer_uncertainties) == 1
    assert "shift" in s2.tracer_uncertainties
    shift2 = s2.tracer_uncertainties["shift"]
    assert shift2.name == "shift"
    assert isinstance(shift2, sacc.NZShiftStretchUncertainty)
    assert shift2.tracer_names == tracer_names
    assert np.allclose(shift2.mean, mu)
    assert np.allclose(shift2.linear_transformation, np.diag(chol))

def test_nzshift_stretch_saving():
    s = sacc.Sacc()
    z = np.linspace(0, 1, 100)
    nz1 = np.exp(-((z - 0.3) ** 2) / (0.1 ** 2))
    nz2 = np.exp(-((z - 0.6) ** 2) / (0.1 ** 2))
    s.add_tracer('NZ', 'source1', z, nz1)
    s.add_tracer('NZ', 'source2', z, nz2)

    tracer_names = ["source1", "source2"]
    mu = [0.001, 0.002]
    # now test with a filled matrix
    chol = np.array([[0.01, 0], [0, 0.02]])
    shift = sacc.NZShiftStretchUncertainty("shift", tracer_names, mu, chol)
    s.add_tracer_uncertainty_object(shift)


    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_nzshift.sacc')
        s.save_fits(filename)
        s2 = sacc.Sacc.load_fits(filename)

    assert len(s2.tracer_uncertainties) == 1
    assert "shift" in s2.tracer_uncertainties
    shift2 = s2.tracer_uncertainties["shift"]
    assert shift2.name == "shift"
    assert isinstance(shift2, sacc.NZShiftStretchUncertainty)
    assert shift2.tracer_names == tracer_names
    assert np.allclose(shift2.mean, mu)
    assert np.allclose(shift2.linear_transformation, chol)

def test_nzlinear_uncertainty_saving():
    s = sacc.Sacc()
    z = np.linspace(0, 1, 100)
    nz1 = np.exp(-((z - 0.3) ** 2) / (0.1 ** 2))
    nz2 = np.exp(-((z - 0.6) ** 2) / (0.1 ** 2))
    s.add_tracer('NZ', 'source1', z, nz1)
    s.add_tracer('NZ', 'source2', z, nz2)

    tracer_names = ["source1", "source2"]
    mean = [0.0, -0.001]
    sigma = [0.01, 0.02]
    linear = sacc.NZLinearUncertainty("linear", tracer_names, mean, sigma)
    s.add_tracer_uncertainty_object(linear)


    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_nzlinear.sacc')
        s.save_fits(filename)
        s2 = sacc.Sacc.load_fits(filename)

    assert len(s2.tracer_uncertainties) == 1
    assert "linear" in s2.tracer_uncertainties
    linear2 = s2.tracer_uncertainties["linear"]
    assert linear2.name == "linear"
    assert isinstance(linear2, sacc.NZLinearUncertainty)
    assert linear2.tracer_names == tracer_names
    assert np.allclose(linear2.mean, mean)
    assert np.allclose(linear2.linear_transformation, np.diag(sigma))

def test_equality_mismatched_covariance(filled_sacc):
    x = filled_sacc.copy()
    cov = make_full_cov(len(filled_sacc))
    x.add_covariance(cov.copy())
    assert x == x

    y = filled_sacc.copy()
    assert x != y

def test_equality_mismatched_metadata(filled_sacc):
    x = filled_sacc.copy()
    x.metadata['test'] = 'value'
    assert x == x

    y = filled_sacc.copy()
    assert x != y

def test_covariance_io(filled_sacc):
    s = filled_sacc.copy()
    cov = make_full_cov(len(filled_sacc))
    s.add_covariance(cov.copy())
    with tempfile.TemporaryDirectory() as tmpdir:
        fits_filename = os.path.join(tmpdir, 'test_covariance_io.sacc')
        hdf_filename = os.path.join(tmpdir, 'test_covariance_io.hdf5')
        s.save_fits(fits_filename)
        s.save_hdf5(hdf_filename)
        s2 = sacc.Sacc.load_fits(fits_filename)
        s3 = sacc.Sacc.load_hdf5(hdf_filename)
        assert s2 == s
        assert s3 == s

if __name__ == "__main__":
    test_bandpower_window()
