from copy import deepcopy
import re
import pytest
import numpy as np

import sacc.covariance as cov

@pytest.fixture
def full_cov() -> cov.FullCovariance:
    x = np.random.uniform(size=(3, 3))
    return cov.FullCovariance((x + x.T)/2)

@pytest.fixture
def diagonal_cov() -> cov.DiagonalCovariance:
    x = np.random.uniform(size=(4))
    return cov.DiagonalCovariance(x)

@pytest.fixture
def block_diagonal_cov() -> cov.BlockDiagonalCovariance:
    block1 = np.random.uniform(size=(3, 3))
    block1 = (block1 + block1.T)/2

    block2 = np.random.uniform(size=(1, 1))

    block3 = np.random.uniform(size=(4, 4))
    block3 = (block3 + block3.T)/2

    return cov.BlockDiagonalCovariance((block1, block2, block3))

def test_full_covariance_equality(full_cov: cov.FullCovariance,
                                  diagonal_cov: cov.DiagonalCovariance,
                                  block_diagonal_cov: cov.BlockDiagonalCovariance):
    x = np.random.uniform(size=(5, 5))
    c1 = cov.FullCovariance((x + x.T)/2)
    c2 = cov.FullCovariance((x + x.T)/2)
    assert c1 is not c2
    assert c1.covmat is not c2.covmat  # should be a copy
    assert np.allclose(c1.covmat, c2.covmat)  # should an *equal* copy
    assert c1 == c2

    assert c1 != full_cov
    assert c1 != diagonal_cov
    assert c1 != block_diagonal_cov
    assert c2 != 10

def test_diagonal_covariance_equality(full_cov: cov.FullCovariance,
                                      diagonal_cov: cov.DiagonalCovariance,
                                      block_diagonal_cov: cov.BlockDiagonalCovariance):
    x = np.random.uniform(size=(5))
    c1 = cov.DiagonalCovariance(x)
    c2 = cov.DiagonalCovariance(x.copy())
    assert c1 is not c2
    assert c1.diag is not c2.diag  # should be a copy
    assert np.allclose(c1.diag, c2.diag)  # should an *equal* copy
    assert c1 == c2

    assert c1 != full_cov
    assert c1 != diagonal_cov
    assert c1 != block_diagonal_cov
    assert c2 != 10

def test_block_diagonal_covariance_equality(full_cov: cov.FullCovariance,
                                            diagonal_cov: cov.DiagonalCovariance,
                                            block_diagonal_cov: cov.BlockDiagonalCovariance):
    blocks = [np.random.uniform(size=(3, 3)), np.random.uniform(size=(2, 2)), np.random.uniform(size=(4, 4))]
    c1 = cov.BlockDiagonalCovariance(blocks)
    c2 = cov.BlockDiagonalCovariance(deepcopy(blocks))
    assert c1 is not c2
    assert c1.blocks is not c2.blocks  # should be a copy
    for b1, b2 in zip(c1.blocks, c2.blocks):
        assert b1 is not b2
        assert np.allclose(b1, b2)
    assert c1 == c2

    assert c1 != full_cov
    assert c1 != diagonal_cov
    assert c1 != block_diagonal_cov
    assert c2 != 10


def test_full_covariance_tables(full_cov: cov.FullCovariance):
    table = full_cov.to_table()
    recovered = cov.FullCovariance.from_table(table)
    assert recovered is not full_cov
    assert recovered == full_cov

def test_diagonal_covariance_tables(diagonal_cov: cov.DiagonalCovariance):
    table = diagonal_cov.to_table()
    recovered = cov.DiagonalCovariance.from_table(table)
    assert recovered is not diagonal_cov
    assert recovered == diagonal_cov

def test_block_diagonal_covariance_tables(block_diagonal_cov: cov.BlockDiagonalCovariance):
    tables = block_diagonal_cov.to_tables()
    assert len(tables) == 3
    recovered = cov.BlockDiagonalCovariance.from_tables(tables)
    assert recovered is not block_diagonal_cov
    assert recovered == block_diagonal_cov

def test_make_full():
    x = np.random.uniform(size=(3,3))
    m = (x + x.T)/2
    c = cov.BaseCovariance.make(m)
    assert isinstance(c, cov.FullCovariance)
    assert np.allclose(c.covmat, m)

def test_make_diag():
    # We use a 2-d array with one dimension set to 1 so that the squeeze in the
    # implementation is used.
    x = np.random.uniform(size=(5, 1))
    c = cov.BaseCovariance.make(x)
    assert isinstance(c, cov.DiagonalCovariance)
    assert np.allclose(c.diag, x.squeeze())
    assert c.diag.shape == (5,)

    x = np.array(5.5)
    assert x.ndim == 0
    c = cov.BaseCovariance.make(x)
    assert isinstance(c, cov.DiagonalCovariance)
    assert np.allclose(c.diag, x)
    assert c.diag.shape == (1,)



def test_make_blocks():
    blocks = [np.random.uniform(size=(3,3)), np.random.uniform(size=(2,2)), np.random.uniform(size=(4,4)), 12.0]
    c = cov.BaseCovariance.make(blocks)
    assert isinstance(c, cov.BlockDiagonalCovariance)
    for b1, b2 in zip(c.blocks, blocks):
        assert isinstance(b1, np.ndarray)
        assert len(b1.shape) == 2
        assert b1.shape[0] == b1.shape[1]
        assert np.allclose(b1, b2)

def test_make_raises():
    bad_matrix = np.random.uniform(size=(2, 3, 2))
    with pytest.raises(ValueError, match=re.escape("Covariance is not a 2D square matrix - shape: (2, 3, 2)")):
        _ = cov.BaseCovariance.make(bad_matrix)
    bad_list = [10, bad_matrix, 12.0]
    with pytest.raises(ValueError, match=re.escape("Covariance block has wrong size or shape (2, 3, 2)")):
        _ = cov.BaseCovariance.make(bad_list)
