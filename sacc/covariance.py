from astropy.table import Table, Column
import scipy.linalg
import numpy as np
import warnings

from .utils import invert_spd_matrix
from .io import BaseIO, ONE_OBJECT_PER_TABLE, ONE_OBJECT_MULTIPLE_TABLES

class BaseCovariance(BaseIO):
    """
    The abstract base class for covariances in different forms.
    These are not currently designed to be modified after creation.

    The three concrete subclasses that are created are:

    FullCovariance - for dense matrices

    BlockDiagonalCovariance - for block diagonal matrices
        (those in which some sub-blocks are dense but without correlation
        between the blocks

    DiagonalCovariance - a covariance where the elements are uncorrelated

    Attributes
    ----------
    cov_type: string
        The type of the covariance (class variable)
    """
    _sub_classes = {}

    def __init__(self):
        """Abstract superclass constructor.

        All the subclasses need _dense and _dense_inverse forms.

        """
        self._dense = None
        self._dense_inverse = None

        # At the moment we only allow one covariance object per table,
        # so this is only used for consistency when saving objects.
        self.name = "cov"

    def __eq__(self, other):
        """
        Test for equality

        Parameters
        ----------
        other: object
            The other object to test for equality

        Returns
        -------
        equal: bool
            True if the objects are equal
        """
        if not isinstance(other, self.__class__):
            return False
        # We do not test the inverse; we rely on the fact that
        # if the dense matrices are equal, then the inverses will be equal.
        # We are also relying on each subclass to have an instance variable
        # 'size'.
        return self.name == other.name and \
            self.size == other.size and \
            ((self._dense is None and other._dense is None) or \
            np.allclose(self._dense, other._dense))

    @classmethod
    def from_hdu(cls, hdu):
        """
        Make a covariance object from an astropy FITS HDU object.

        The type of the covariance is determined from a keyword
        in the HDU, and then the corresponding subclass from_hdu
        method is called.

        Parameters
        ----------
        hdu: astropy.fits.ImageHDU instance
            An HDU object with covariance info in it

        Returns
        -------
        instance: BaseCovariance
            A covariance instance
        """
        warnings.warn("You are using an older SACC legacy SACC file format with old covariance data."
                      " Consider updating it by loading it and saving it again with the latest SACC version.",
                      DeprecationWarning)
        subclass_name = hdu.header['saccclss']
        subclass = cls._sub_classes[subclass_name]
        return subclass.from_hdu(hdu)

    @classmethod
    def make(cls, cov):
        """Make an appropriate covariance object from the matrix info itself.

        You can pass in a list of covariance blocks for a block-diagonal,
        covariance a 1D array for a diagonal covariance, or a full matrix.

        A different subclass is returned for each of these cases.

        Parameters
        ----------
        cov: list[array] or array
            If a list, it should be a list of array-like objects each of which
            can be coerced into a 2d array. A BlockDiagonalCovariance will be
            returned.

            If an array, it should be either 1D or 2d and square. Either a
            DiagonalCovariance or a FullCovariance will be returned.
        """
        if isinstance(cov, list):
            for block in cov:
                block = np.atleast_2d(block)
                if (block.ndim != 2) or (block.shape[0] != block.shape[1]):
                    raise ValueError("Covariance block has wrong size "
                                     f"or shape {block.shape}")
            return BlockDiagonalCovariance(cov)
        cov = np.array(cov).squeeze()
        if cov.ndim == 0:
            return DiagonalCovariance(np.atleast_1d(cov))
        if cov.ndim == 1:
            return DiagonalCovariance(cov)
        if (cov.ndim != 2) or (cov.shape[0] != cov.shape[1]):
            raise ValueError("Covariance is not a 2D square matrix "
                             f"- shape: {cov.shape}")
        return FullCovariance(cov)

    @property
    def dense(self):
        """
        A dense matrix form of the covariance

        Parameters
        ----------
        None

        Returns
        -------
        covmat: 2D array
            Numpy array of dense form of matrix
        """
        if self._dense is None:
            self._dense = self._get_dense()

        return self._dense

    @property
    def inverse(self):
        """A dense matrix form of the inverse of the covariance matrix

        Returns
        -------
        invC: array
            Inverse covariance
        """
        if self._dense_inverse is None:
            self._dense_inverse = self._get_dense_inverse()
        return self._dense_inverse


class FullCovariance(BaseCovariance, type_name='full'):
    """
    A covariance subclass representing a full matrix with correlations
    anywhere.  Represented as an n x n matrix.

    Attributes
    ----------
    size: int
        the length of the corresponding data vector

    covmat: 2D array
        The matrix itself, of shape (size x size)
    """

    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(self, covmat):
        self.covmat = np.atleast_2d(covmat)
        self.size = self.covmat.shape[0]
        super().__init__()

    def __eq__(self, other):
        return super().__eq__(other) and \
            np.allclose(self.covmat, other.covmat)

    @classmethod
    def from_hdu(cls, hdu):
        """

        Load a covariance object from the data in the HDU
        LEGACY METHOD: new sacc files will use from_table

        Parameters
        ----------
        hdu: astropy.fits.ImageHDU instance

        Returns
        -------
        cov: FullCovariance
            Loaded covariance object
        """
        C = hdu.data
        return cls(C)

    def to_table(self):
        """
        Make an astropy table object with this covariance in it.

        Parameters
        ----------
        None

        Returns
        -------
        table: astropy.table.Table instance
            Table that can be used to reconstruct the object.
        """
        # Store as a single vector column ('row') to avoid FITS TFIELDS>999
        # Each table row is one row of the covariance matrix
        table = Table([Column(name='row', data=self.covmat)])
        table.meta['SIZE'] = self.size
        return table

    @classmethod
    def from_table(cls, table):
        """
        Load a covariance object from the data in the table

        Parameters
        ----------
        table: astropy.table.Table instance

        Returns
        -------
        cov: FullCovariance
            Loaded covariance object
        """
        size = table.meta['SIZE']
        # Support both legacy many-column format and new single-column format
        if 'row' in table.colnames:
            covmat = np.array(list(table['row']))
        else:
            covmat = np.array([table[f'col_{i}'] for i in range(size)])
        return cls(covmat)


    def keeping_indices(self, indices):
        """
        Return a new instance with only the specified indices retained.

        Parameters
        ----------
        indices: array or list
            Either an array or list of integer indices, or a boolean
            array of the same size (1D) as the matrix.
            Specifies rows/cols to keep in the new matrix.

        Returns
        -------
        cov: FullCovariance
            A covariance with only the corresponding data points remaining
        """
        C = self.covmat[indices][:, indices]
        return self.__class__(C)

    def get_block(self, indices):
        """Read a (not necessarily contiguous) sublock of the matrix

        Parameters
        ----------
        indices: array
            An array of integer indices

        Returns
        -------
        block: array
            a 2D array of the relevant sub-block of the matrix
        """
        return self.covmat[indices][:, indices]

    def _get_dense_inverse(self):
        return invert_spd_matrix(self.covmat)

    def _get_dense(self):
        # Internal method to get a dense form of the matrix.
        # Use the property Covariance.dense instead of calling this
        # directly.
        return self.covmat.copy()


class BlockDiagonalCovariance(BaseCovariance, type_name='block'):
    """A covariance subclass representing block diagonal covariances

    Block diagonal covariances have sub-blocks that are full dense matrices,
    but without correlations between the blocks. This feature can be taken
    advantage of when doing matrix operations like multiplication or inversion.

    Parameters
    ----------
    blocks: list[arrays]
        list of sub-blocks of the matrix

    block_sizes: list[int]
        list of sizes n of each the n x n sub-blocks

    size: int
        overall total size of the matrix
    """

    storage_type = ONE_OBJECT_MULTIPLE_TABLES

    def __init__(self, blocks):
        """Create a BlockDiagonalCovariance object from a list of blocks

        Parameters
        ----------
        blocks: sequence of arrays
            List or other sequence of the sub-matrices
        """
        self.blocks = [np.atleast_2d(B) for B in blocks]
        self.block_sizes = [len(B) for B in self.blocks]
        self.size = sum(self.block_sizes)
        super().__init__()

    def __eq__(self, other):
        return super().__eq__(other) and \
            self.block_sizes == other.block_sizes and \
            all(np.allclose(b1, b2)
                for b1, b2
                in zip(self.blocks, other.blocks))

    @classmethod
    def from_hdu(cls, hdu):
        """Read a covariance object from a loaded FITS HDU.
        LEGACY METHOD: new sacc files will use from_tables

        Parameters
        ----------
        hdu: FITS HDU object as read in by astropy.

        Returns
        -------
        cov: BlockDiagonalCovariance
            Loaded covariance object
        """
        n = hdu.header['blocks']
        block_sizes = [hdu.header[f'size_{i}'] for i in range(n)]
        s = 0

        blocks = []
        for b in block_sizes:
            B = hdu.data[s:s + b**2].reshape((b, b))
            s += b**2
            blocks.append(B)
        return cls(blocks)

    @classmethod
    def from_tables(cls, tables):
        """
        Load a covariance object from the data in the tables

        Parameters
        ----------
        tables: list
            list of astropy.table.Table instances in block order

        Returns
        -------
        cov: BlockDiagonalCovariance
            Loaded covariance object
        """

        blocks = []
        # Get the block count from the first table
        nblock = list(tables.values())[0].meta['SACCBCNT']
        for i in range(nblock):
            table = tables[f'block_{i}']
            block_size = table.meta['SACCBSZE']
            if 'block_row' in table.colnames:
                block = np.array(list(table['block_row']))
            else:
                cols = [table[f'block_col_{j}'] for j in range(block_size)]
                block = np.array(cols)
            blocks.append(block)
        return cls(blocks)

    def to_tables(self):
        """
        Make an astropy table object with this covariance in it.

        Parameters
        ----------
        None

        Returns
        -------
        table: astropy.table.Table instance
            Table that can be used to reconstruct the object.
        """
        tables = {}
        nblock = len(self.blocks)
        for j, block in enumerate(self.blocks):
            b = len(block)
            # Use single vector column to minimize TFIELDS
            table = Table([Column(name='block_row', data=block)])
            table.meta['SIZE'] = self.size
            table.meta['SACCBIDX'] = j
            table.meta['SACCBCNT'] = nblock
            table.meta['SACCBSZE'] = b
            tables[f'block_{j}'] = table
        return tables

    def get_block(self, indices):
        """Read a (not necessarily contiguous) sublock of the matrix

        Parameters
        ----------
        indices: array
            An array of integer indices, which must be in
            ascending order

        Returns
        -------
        cov: array
            A full (dense) 2x2 array of the submatrix.
        """
        indices = np.array(indices)

        if np.any(np.diff(indices)) < 0:
            raise ValueError("Indices passed to "
                             "BlockDiagonalCovariance.get_block "
                             "must be in ascending order")
        s = 0
        sub_blocks = []
        for block, sz in zip(self.blocks, self.block_sizes):
            e = s + sz
            m = indices[(indices >= s) & (indices < e)] - s
            sub_blocks.append(block[m][:, m])
            s += sz
        return scipy.linalg.block_diag(*sub_blocks)

    def keeping_indices(self, indices):
        """
        Return a new instance with only the specified elements retained.

        This method will try to return another BlockDiagonalCovariance if
        it can, but otherwise will revert to a full one: if the mask passed
        in is of a boolean type or if it is integers in it can remain
        block diagonal

        Parameters
        ----------
        indices: array or list
            Either an array or list of integer indices, or a boolean
            array of the same size (1D) as the matrix.
            Specifies rows/cols to keep in the new matrix.

        Returns
        -------
        cov: FullCovariance or BlockDiagonalCovariance
            A covariance with only the corresponding data points remaining
        """
        indices = np.array(indices)

        if indices.dtype == bool:
            breaks = np.cumsum(self.block_sizes)[:-1]
            block_masks = np.split(indices, breaks)
            blocks = [self.blocks[i][m][:, m] for i, m in
                      enumerate(block_masks)]
            return self.__class__(blocks)
        if (np.diff(indices) > 0).all():
            s = 0
            sub_blocks = []
            for block, sz in zip(self.blocks, self.block_sizes):
                e = s + sz
                m = indices[(indices >= s) & (indices < e)] - s
                sub_blocks.append(block[m][:, m])
                s += sz
            return self.__class__(sub_blocks)
        C = scipy.linalg.block_diag(*self.blocks)
        C = C[indices][:, indices]
        return FullCovariance(C)

    def _get_dense_inverse(self):
        # Invert all the blocks individually and then
        # connect them all together
        return scipy.linalg.block_diag(*[invert_spd_matrix(B)
                                         for B in self.blocks])

    def _get_dense(self):
        # Internal method to get a dense form of the matrix.
        # Use the property Covariance.dense instead of calling this
        # directly.
        return scipy.linalg.block_diag(*self.blocks)


class DiagonalCovariance(BaseCovariance, type_name='diagonal'):
    """A covariance subclass representing covariances that are
    purely diagonal.

    Parameters
    ----------
    size: int
        The size of the matrix

    diag: array
        The diagonal terms in the covariance (i.e. the variances)
    """

    storage_type = ONE_OBJECT_PER_TABLE

    def __init__(self, variances):
        """
        Create a DiagonalCovariance object from the variances
        of the data points.

        Parameters
        ----------
        variances: array
            2D array of variances of the data points.
        """
        self.diag = np.atleast_1d(variances)
        self.size = len(self.diag)
        super().__init__()

    def __eq__(self, other):
        return super().__eq__(other) and \
            np.allclose(self.diag, other.diag)


    def keeping_indices(self, indices):
        """
        Return a new DiagonalCovariance with only the specified indices
        retained.

        Parameters
        ----------
        indices: array or list
            Either an array or list of integer indices, or a boolean
            array of the same size (1D) as the matrix.
            Specifies rows/cols to keep in the new matrix.

        Returns
        -------
        cov: DiagonalCovariance
            A covariance with only the corresponding data points remaining
        """
        D = self.diag[indices]
        return self.__class__(D)

    @classmethod
    def from_table(cls, table):
        """
        Load a covariance object from the data in the table

        Parameters
        ----------
        table: astropy.table.Table instance

        Returns
        -------
        cov: DiagonalCovariance
            Loaded covariance object
        """
        D = table['variance']
        return cls(D)

    def to_table(self):
        """
        Make an astropy table object with this covariance in it.

        Parameters
        ----------
        None

        Returns
        -------
        table: astropy.table.Table instance
            Table that can be used to reconstruct the object.
        """
        table = Table(data=[self.diag], names=['variance'])
        table.meta['SIZE'] = self.size
        return table

    @classmethod
    def from_hdu(cls, hdu):
        """
        Load a covariance object from the data in the HDU

        Parameters
        ----------
        hdu: astropy.fits.BinTableHDU instance

        Returns
        -------
        cov: DiagonalCovariance
            Loaded covariance object
        """
        D = hdu.data['variance']
        return cls(D)

    def get_block(self, indices):
        """Read a (not necessarily contiguous) sublock of the matrix

        Parameters
        ----------
        indices: array
            An array of integer indices, which should be in
            ascending order (for consistency with the
            block diagonal interface)

        Returns
        -------
        cov: array
            A full (dense) 2x2 array of the submatrix.
        """
        return np.diag(self.diag[indices])

    def _get_dense_inverse(self):
        # Trivial inverse
        return np.diag(1.0/self.diag)

    def _get_dense(self):
        # Internal method to get a dense form of the matrix.
        # Use the property Covariance.dense instead of calling this
        # directly.
        return np.diag(self.diag)


def concatenate_covariances(*covariances):
    # If all the covariances are diagonal then the concatenated
    # version can be diagonal
    if all(isinstance(cov, DiagonalCovariance) for cov in covariances):
        variances = np.concatenate([cov.diag for cov in covariances])
        return DiagonalCovariance(variances)

    # Otherwise we have to get things in a common form, and
    # make a block-diagonal covariance.
    blocks = []

    # For each of the pieces we extract any blocks
    # that will go into the concatenation
    for cov in covariances:
        # For an existing block-diagonal covariance
        # we retain the block structure
        if isinstance(cov, BlockDiagonalCovariance):
            blocks += cov.blocks
        # For everything else we just use a dense matrix
        else:
            blocks.append(cov.dense)

    return BlockDiagonalCovariance(blocks)
