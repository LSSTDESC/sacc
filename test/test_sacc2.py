import sacc
import sacc.data_types
import numpy as np
import scipy.linalg

def test_construct():
    s = sacc.Sacc()

    # Tracer
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2
    s.add_tracer('NZ', 'source_0', z, nz)

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0', 'source_0')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee, tracers, ee, ell=10.0*i)


def test_full_cov():
    covmat = np.random.uniform(size=(100,100))
    C = sacc.covariance.BaseCovariance.make(covmat)
    assert C.size == 100
    assert isinstance(C, sacc.covariance.FullCovariance)
    assert np.all(C.covmat==covmat)
    hdu = C.to_hdu()
    C2 = sacc.covariance.BaseCovariance.from_hdu(hdu)
    assert np.allclose(C.covmat, C2.covmat)


def test_block_cov():
    covmat = [np.random.uniform(size=(50,50)), np.random.uniform(size=(100,100)), np.random.uniform(size=(150,150))]
    C = sacc.covariance.BaseCovariance.make(covmat)
    assert C.size == 300
    assert isinstance(C, sacc.covariance.BlockDiagonalCovariance)
    hdu = C.to_hdu()
    C2 = sacc.covariance.BaseCovariance.from_hdu(hdu)
    assert len(C2.blocks)==3
    assert C.block_sizes == C2.block_sizes
    for i in range(3):
        assert np.allclose(C.blocks[i], C2.blocks[i])


def test_misc_tracer():
    md1 = {'potato': 'if_necessary', 'answer': 42, 'height':1.83}
    md2 = {'potato': 'never'}
    T1 = sacc.BaseTracer.make('misc', 'tracer1', metadata=md1)
    T2 = sacc.BaseTracer.make('misc', 'tracer2', metadata=md2)
    assert T1.metadata == md1
    assert T2.metadata == md2

    tables = sacc.BaseTracer.to_tables([T1, T2])
    D = sacc.BaseTracer.from_tables(tables)

    T1a = D['tracer1']
    T2a = D['tracer2']
    assert T1a.metadata == md1
    assert T2a.metadata == md2

def test_nz_tracer():
    md1 = {'potato': 'if_necessary', 'answer': 42, 'height':1.83}
    md2 = {'potato': 'never'}
    z = np.arange(0., 1., 0.01)
    Nz1 = 1*z # not a sensible N(z)!
    Nz2 = 2*z # not a sensible N(z)!

    Nz3 = 3*z
    Nz4 = 4*z

    more_nz = {'v1':Nz3, 'v2':Nz4}

    T1 = sacc.BaseTracer.make('NZ', 'tracer1', z, Nz1, extra_columns=more_nz, metadata=md1)
    T2 = sacc.BaseTracer.make('NZ', 'tracer2', z, Nz2, metadata=md2)
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
    Nz1 = 1*z # not a sensible N(z)!
    Nz2 = 2*z
    T1 = sacc.BaseTracer.make('NZ', 'tracer1', z, Nz1)
    T2 = sacc.BaseTracer.make('NZ', 'tracer2', z, Nz2, metadata=md1)

    M1 = sacc.BaseTracer.make("misc", "sample1", metadata=md2)
    M2 = sacc.BaseTracer.make("misc", "sample2", metadata=md3)

    tables = sacc.BaseTracer.to_tables([T1, M1, T2, M2])
    recovered = sacc.BaseTracer.from_tables(tables)
    assert recovered['sample1'].metadata['rank']=='duke'
    assert recovered['sample2'].metadata['robes']==78
    assert np.all(recovered['tracer1'].nz == Nz1)
    assert recovered['tracer2'].metadata['potato']=='never'

def test_inverses():
    N = 25
    C = np.random.uniform(0,1, size=(N,N))
    C = (C+C.T) + np.eye(N)*20
    M1 = sacc.BaseCovariance.make(C)
    assert M1.size == N
    invC = M1.inverted()
    I = np.dot(invC, C)
    assert np.allclose(I, np.eye(N))

    blocks = [np.random.uniform(0,1, size=(5,5)) for i in range(5)]
    for b in blocks:
        b += b.T + np.eye(5)*20

    M2 = sacc.BaseCovariance.make(blocks)
    assert M2.size == N
    M2dense = np.zeros((N,N))
    for i in range(5):
        M2dense[i*5:i*5+5,i*5:i*5+5] = blocks[i]
    invC2 = M2.inverted()
    I = np.dot(invC2, M2dense)
    assert np.allclose(I, np.eye(N))

    d = abs(np.random.uniform(0,1,size=N))+1
    M3 = sacc.BaseCovariance.make(d)
    assert M3.size == N
    invC3 = M3.inverted()
    assert np.count_nonzero(invC3 - np.diag(np.diagonal(invC3)))==0
    assert np.allclose(invC3.diagonal() * d, 1)


def test_data_point():
    from sacc.data_types import DataPoint
    dt = sacc.data_types.standard_types.galaxy_shearDensity_cl_e
    value = 13.4
    tracers = ('aaa', 'bbb')
    tags = {'ell':12, 'theta':14.3}
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
    s.add_tracer('NZ', 'source_1', z, nz)
    s.add_tracer('NZ', 'source_2', z, nz)

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0', 'source_0')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee, tracers, ee, ell=10.0*i)
    for i in range(20):
        bb = 0.1 * i
        tracers = ('source_1', 'source_1')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_bb, tracers, bb, ell=10.0*i)
    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_2', 'source_2')
        s.add_data_point(sacc.standard_types.galaxy_shear_cl_ee, tracers, ee, ell=10.0*i)

    # Select by data type
    s2 = s.copy()
    s2.keep_selection(data_type=sacc.standard_types.galaxy_shear_cl_bb)
    assert all(d.data_type==sacc.standard_types.galaxy_shear_cl_bb for d in s2.data)
    assert len(s2)==20

    # From multiple tracers
    s2 = s.copy()
    s2.keep_selection(data_type=sacc.standard_types.galaxy_shear_cl_ee)
    assert all(d.data_type==sacc.standard_types.galaxy_shear_cl_ee for d in s2.data)
    assert len(s2)==40

    # Test removing a single tracer
    s2 = s.copy()
    s2.remove_selection(tracers=('source_1', 'source_1'))
    for i,d in enumerate(s2.data):
        if i<20:
            assert d.tracers == ('source_0', 'source_0')
        else:
            assert d.tracers == ('source_2', 'source_2')
    assert all(d.data_type==sacc.standard_types.galaxy_shear_cl_ee for d in s2.data)
    assert len(s2)==40

    # Test selecting by tag
    s2 = s.copy()
    s2.remove_selection(ell__lt=55)
    ell = s2.get_tag('ell')
    for e in ell:
        assert e>55
    s2 = s.copy()
    s2.keep_selection(ell__lt=55)
    ell = s2.get_tag('ell')
    for e in ell:
        assert e<55

    # Cutting just by index
    s2 = s.copy()
    ind = s2.indices(tracers=('source_1', 'source_1'))
    assert (ind == np.arange(20, 40)).all()

    # multiple selections
    s2 = s.copy()
    ind = s2.indices(tracers=('source_2', 'source_2'), ell__lt=45)
    assert len(ind)==5

def test_cutting_block_cov():
    covmat = [np.random.uniform(size=(50,50)), np.random.uniform(size=(100,100)), np.random.uniform(size=(150,150))]
    C = sacc.covariance.BaseCovariance.make(covmat)
    ind = list(range(50))
    C2 = C.keeping_indices(np.arange(50))
    assert C2.size == len(ind)
    assert np.allclose(C2.get_block(ind), covmat[0])

def test_cutting_full_cov():
    covmat = np.random.uniform(size=(100,100))
    C = sacc.covariance.BaseCovariance.make(covmat)
    ind = np.arange(10, dtype=int)
    C2 = C.keeping_indices(ind)
    assert np.allclose(C2.get_block(ind), covmat[:10,:10])

def test_cutting_diag_cov():
    diag = np.random.uniform(size=(100,))
    C = sacc.covariance.BaseCovariance.make(diag)
    ind = np.arange(20, dtype=int)
    C2 = C.keeping_indices(ind)
    assert np.allclose(C2.get_block(ind).diagonal(), diag[:20])

def test_parse_data_names():
    for name in sacc.data_types.required_tags:
        sources, props, stat, sub = sacc.parse_data_type_name(name)
        name2 = sacc.build_data_type_name(sources, props, stat, sub)
        assert name == name2