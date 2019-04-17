import sacc2
import numpy as np


def test_construct():
    s = sacc2.Sacc()

    # Tracer
    z = np.arange(0., 1.0, 0.01)
    nz = (z-0.5)**2/0.1**2
    s.add_tracer('NZ', 'source_0', z, nz)

    for i in range(20):
        ee = 0.1 * i
        tracers = ('source_0', 'source_0')
        s.add_data_point(sacc2.known_types.galaxy_shear_ee, tracers, ee, ell=10.0*i)


def test_full_cov():
    covmat = np.random.uniform(size=(100,100))
    C = sacc2.covariance.BaseCovariance.make(covmat, 100)
    assert isinstance(C, sacc2.covariance.FullCovariance)
    assert np.all(C.covmat==covmat)
    hdu = C.to_hdu()
    C2 = sacc2.covariance.BaseCovariance.from_hdu(hdu)
    assert np.allclose(C.covmat, C2.covmat)


def test_block_cov():
    covmat = [np.random.uniform(size=(50,50)), np.random.uniform(size=(100,100)), np.random.uniform(size=(150,150))]
    C = sacc2.covariance.BaseCovariance.make(covmat, 300)
    assert isinstance(C, sacc2.covariance.BlockDiagonalCovariance)
    hdu = C.to_hdu()
    C2 = sacc2.covariance.BaseCovariance.from_hdu(hdu)
    assert len(C2.blocks)==3
    assert C.block_sizes == C2.block_sizes
    for i in range(3):
        assert np.allclose(C.blocks[i], C2.blocks[i])


def test_misc_tracer():
    md1 = {'potato': 'if_necessary', 'answer': 42, 'height':1.83}
    md2 = {'potato': 'never'}
    T1 = sacc2.BaseTracer.make('misc', 'tracer1', metadata=md1)
    T2 = sacc2.BaseTracer.make('misc', 'tracer2', metadata=md2)
    assert T1.metadata == md1
    assert T2.metadata == md2

    tables = sacc2.BaseTracer.to_tables([T1, T2])
    D = sacc2.BaseTracer.from_tables(tables)

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

    T1 = sacc2.BaseTracer.make('NZ', 'tracer1', z, Nz1, more_nz=more_nz, metadata=md1)
    T2 = sacc2.BaseTracer.make('NZ', 'tracer2', z, Nz2, metadata=md2)
    assert T1.metadata == md1
    assert T2.metadata == md2
    
    tables = sacc2.BaseTracer.to_tables([T1, T2])
    D = sacc2.BaseTracer.from_tables(tables)

    T1a = D['tracer1']
    T2a = D['tracer2']
    assert T1a.metadata == md1
    assert T2a.metadata == md2

    assert np.all(T1a.more_nz['v1'] == Nz3)
    assert np.all(T1a.more_nz['v2'] == Nz4)

def test_mixed_tracers():
    md1 = {'potato': 'never'}
    md2 = {'rank': 'duke'}
    md3 = {'rank': 'earl', 'robes': 78}
    z = np.arange(0., 1., 0.01)
    Nz1 = 1*z # not a sensible N(z)!
    Nz2 = 2*z
    T1 = sacc2.BaseTracer.make('NZ', 'tracer1', z, Nz1)
    T2 = sacc2.BaseTracer.make('NZ', 'tracer2', z, Nz2, metadata=md1)

    M1 = sacc2.BaseTracer.make("misc", "sample1", metadata=md2)
    M2 = sacc2.BaseTracer.make("misc", "sample2", metadata=md3)

    tables = sacc2.BaseTracer.to_tables([T1, M1, T2, M2])
    recovered = sacc2.BaseTracer.from_tables(tables)
    assert recovered['sample1'].metadata['rank']=='duke'
    assert recovered['sample2'].metadata['robes']==78
    assert np.all(recovered['tracer1'].nz == Nz1)
    assert recovered['tracer2'].metadata['potato']=='never'


if __name__ == '__main__':
    test_save_load_tracer()