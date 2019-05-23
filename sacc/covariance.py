from astropy.io import fits
from astropy.table import Table
import scipy.linalg
import numpy as np

from .utils import invert_spd_matrix

class BaseCovariance:
    """
    The abstract base class for covariances in different forms.

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
    _covariance_classes = {}

    # This method gets called whenever a subclass is
    # defined.  The keyword argument in the class definition
    # (e.g. cov_type='full' below is passed to this class method)
    @classmethod
    def __init_subclass__(cls, cov_type):
        cls._covariance_classes[cov_type] = cls
        cls.cov_type = cov_type

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

        Returns: instance
            A covariance instance
        """
        subclass_name = hdu.header['saccclss']
        subclass = cls._covariance_classes[subclass_name]
        return subclass.from_hdu(hdu)

    @classmethod
    def make(cls, cov):
        """Make an appropriate covariance object from the matrix info itself.

        You can pass in a list of covariance blocks for a block-diagonal, covariance
        a 1D array for a diagonal covariance, or a full matrix.

        A different subclass is returned for each of these cases.

        Parameters
        ----------
        cov: list[array] or array
            If a list, the total length of all the arrays in it
            should equal n.  If an array, it should be either 1D of
            length n or 2D of shape (n x n).

        n: int
            length of the data vector to which this covariance applies
        """
        if isinstance(cov, list):
            s = 0
            for block in cov:
                block = np.atleast_2d(block)
                if (block.ndim != 2) or (block.shape[0] != block.shape[1]):
                    raise ValueError(f"Covariance block has wrong size or shape {block.shape}")
                s += block.shape[0]
            return BlockDiagonalCovariance(cov)
        else:
            cov = np.array(cov).squeeze()
            if cov.ndim == 1:
                return DiagonalCovariance(cov)
            if (cov.ndim != 2) or (cov.shape[0] != cov.shape[1]):
                raise ValueError(f"Covariance is not a 2D square matrix - shape: {cov.shape}")
            return FullCovariance(cov)


class FullCovariance(BaseCovariance, cov_type='full'):
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
    def __init__(self, covmat):
        self.covmat = np.atleast_2d(covmat)
        self.size = self.covmat.shape[0]

    def to_hdu(self):
        """
        Make an astropy FITS HDU object with this covariance in it.
        This is represented as an image.

        Parameters
        ----------
        None

        Returns
        -------
        hdu: astropy.fits.ImageHDU instance
            HDU that can be used to reconstruct the object.
        """
        hdu = fits.ImageHDU(self.covmat)
        hdu.header['EXTNAME'] = 'covariance'
        hdu.header['SACCTYPE'] = 'cov'
        hdu.header['SACCCLSS'] = self.cov_type
        hdu.header['SIZE'] = self.size
        return hdu

    @classmethod
    def from_hdu(cls, hdu):
        """
        Load a covariance object from the data in the HDU

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


    def masked(self, mask):
        """
        Return a new matrix with only the masked elements retained.

        Parameters
        ----------
        mask: array or list
            Either an array or list of integer indices, or a boolean
            array of the same size (1D) as the matrix.
            Specifies rows/cols to keep in the new matrix.

        Returns
        -------

        cov: FullCovariance
            A covariance with only the corresponding data points remaining
        """
        C = self.covmat[mask][:, mask]
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

    def inverted(self):
        """Return the inverse of the covariance matrix, as a dense array.

        Returns
        -------
        invC: array
            Inverse covariance

        """
        return invert_spd_matrix(self.covmat)


class BlockDiagonalCovariance(BaseCovariance, cov_type='block'):
    """A covariance subclass representing block diagonal covariances

    Block diagonal covariances have sub-blocks that are full dense matrices,
    but without correlations between the blocks. This feature can be taken
    advantage of when doing matrix operations like multiplication or inversion.

    Attributes
    ----------
    blocks: list[arrays]
        list of sub-blocks of the matrix

    block_sizes: list[int]
        list of sizes n of each the n x n sub-blocks

    size: int
        overall total size of the matrix

    """
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

    def to_hdu(self):
        """Write a FITS HDU from the data, ready to be saved.

        The data in the HDU is stored as a single 1 x size image,
        and the header contains the information needed to reconstruct it.

        Parameters
        ----------
        None

        Returns
        -------
        hdu: astropy.fits.ImageHDU object
            HDU containing data and metadata
        """
        hdu = fits.ImageHDU(np.concatenate([b.flatten() for b in self.blocks]))
        hdu.name = 'covariance'
        hdu.header['sacctype'] = 'cov'
        hdu.header['saccclss'] = self.cov_type
        hdu.header['size'] = self.size
        hdu.header['blocks'] = len(self.blocks)
        for i, s in enumerate(self.block_sizes):
            hdu.header[f'size_{i}'] = s
        return hdu

    @classmethod
    def from_hdu(cls, hdu):
        """Read a covariance object from a loaded FITS HDU.

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
        if np.any(np.diff(indices))<0:
            raise ValueError("Indices passed to BlockDiagonalCovariance.get_block must be in ascending order")
        n = len(indices)
        s = 0
        sub_blocks = []
        for block, sz in zip(self.blocks, self.block_sizes):
            e = s + sz
            m = indices[(indices >= s) & (indices < e)]
            sub_blocks.append(block[m][:, m])
            s += sz
        return scipy.linalg.block_diag(*sub_blocks)

    def masked(self, mask):
        """
        Return a new matrix with only the masked elements retained.

        This method will try to return another BlockDiagonalCovariance if
        it can, but otherwise will revert to a full one: if the mask passed
        in is of a boolean type or if it is integers in it can remain
        block diagonal

        Parameters
        ----------
        mask: array or list
            Either an array or list of integer indices, or a boolean
            array of the same size (1D) as the matrix.
            Specifies rows/cols to keep in the new matrix.

        Returns
        -------
        cov: FullCovariance or BlockDiagonalCovariance
            A covariance with only the corresponding data points remaining
        """
        if mask.dtype == bool:
            breaks = np.cumsum(self.block_sizes)[:-1]
            block_masks = np.split(mask, breaks)
            blocks = [self.blocks[m][:, m] for m in block_masks]
            return self.__class__(blocks)
        elif (np.diff(mask) > 0).all():
            s = 0
            sub_blocks = []
            for block, sz in zip(self.blocks, self.block_sizes):
                e = s + sz
                m = mask[(mask >= s) & (mask < e)]
                sub_blocks.append(block[m][:, m])
                s += sz
            return self.__class__(sub_blocks)
        else:
            C = scipy.linalg.block_diag(*self.blocks)
            C = C[mask][:, mask]
            return FullCovariance(C)

    def inverted(self):
        """Return the inverse of the covariance matrix, as a dense array.

        Returns
        -------
        invC: array
            Inverse covariance
        """
        return scipy.linalg.block_diag(*[invert_spd_matrix(B) for B in self.blocks])



class DiagonalCovariance(BaseCovariance, cov_type='diagonal'):
    """A covariance subclass representing covariances that are
    purely diagonal.

    Attributes
    ----------
    size: int
        The size of the matrix

    diag: array
        The diagonal terms in the covariance (i.e. the variances)

    """
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

    def to_hdu(self):
        """
        Make an astropy FITS HDU object with this covariance in it.
        In this can a binary table HDU is created.

        Parameters
        ----------
        None

        Returns
        -------
        hdu: astropy.fits.BinTableHDU instance
            HDU that can be used to reconstruct the object.
        """
        table = Table(names=['variance'], data=[self.diag])
        hdu = fits.table_to_hdu(table)
        hdu.name = 'covariance'
        hdu.header['sacctype'] = 'cov'
        hdu.header['saccclss'] = self.cov_type
        return hdu

    def masked(self, mask):
        """
        Return a new DiagonalCovariance with only the masked elements
        retained.

        Parameters
        ----------
        mask: array or list
            Either an array or list of integer indices, or a boolean
            array of the same size (1D) as the matrix.
            Specifies rows/cols to keep in the new matrix.

        Returns
        -------

        cov: DiagonalCovariance
            A covariance with only the corresponding data points remaining
        """
        D = self.diag[mask]
        return self.__class__(D)

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

    def inverted(self):
        """Return the inverse of the covariance matrix, as a dense array.

        Returns
        -------
        invC: array
            Inverse covariance
        """
        return np.diag(1.0/self.diag)
