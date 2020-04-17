import re

from astropy.table import Column
import numpy as np
import scipy.linalg

# These null values are used in place
# of missing values.
null_values = {
    'i': -1437530437530211245,
    'f': -1.4375304375e30,
    'U': '',
}


def hide_null_values(table):
    """Replace None values in an astropy table
    with a null value, and change the column type
    to whatever suits the remaining data points

    Parameters
    ----------
    table: astropy table
        Table to modify in-place
    """

    for name, col in list(table.columns.items()):
        if col.dtype.kind == 'O':
            good_values = [x for x in col if x is not None]
            good_kind = np.array(good_values).dtype.kind
            null = null_values[good_kind]
            good_col = np.array([null if x is None else x for x in col])
            table[name] = Column(good_col)


def remove_dict_null_values(dictionary):
    """Remove values in a dictionary that
    correspond to the null values above.

    Parameters
    ----------
    dictionary: dict
        Dict (or subclass instance or other mapping) to modify in-place
    """
    # will figure out the list of keys to remove
    deletes = []
    for k, v in dictionary.items():
        try:
            dt = np.dtype(type(v)).kind
            if v == null_values[dt]:
                deletes.append(k)
        except (TypeError, KeyError):
            continue
    for d in deletes:
        del dictionary[d]


def unique_list(seq):
    """
    Find the unique elements in a list or other sequence
    while maintaining the order. (i.e., remove any duplicated
    elements but otherwise leave it the same)

    Method from:
    https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order

    Parameters
    ----------
    seq: list or sequence
        Any input object that can be iterated

    Returns
    -------
    L: list
        a new list of the unique objects
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class Namespace:
    """
    This helper class implements a very simple namespace object
    designed to store strings with the same name as their contents
    as a kind of simple string enum.

    N = Namespace('a', 'b', 'c')

    assert N.a=='a'

    assert N['a']=='a'

    assert N.index('b')==1
    """
    def __init__(self, *strings):
        """
        Create the object from a list of strings, which will become attributes
        """
        self._index = {}
        n = 0
        for s in strings:
            self.__dict__[s] = s
            self._index[s] = n
            n += 1

    def __contains__(self, s):
        return hasattr(self, s)

    def __getitem__(self, s):
        return getattr(self, s)

    def __str__(self):
        return "\n".join(f"- {s}" for s in self._index)

    def index(self, s):
        return self._index[s]


def invert_spd_matrix(M, strict=True):
    """
    Invert a symmetric positive definite matrix.

    SPD matrices (for example, covariance matrices) have only
    positive eigenvalues.

    Based on:
    https://stackoverflow.com/questions/40703042/more-efficient-way-to-invert-a-matrix-knowing-it-is-symmetric-and-positive-semi

    Parameters
    ----------
    M: 2d array
        Matrix to invert

    strict: bool, default=True
        If True, require that the matrix is SPD.
        If False, use a slower algorithm that will work on other matrices

    Returns
    -------
    invM: 2d array
        Inverse matrix

    """
    M = np.atleast_2d(M)

    # In strict mode we use a method which will only work for SPD matrices,
    # and raise an exception otherwise.
    if strict:
        L, _ = scipy.linalg.lapack.dpotrf(M, False, False)
        invM, info = scipy.linalg.lapack.dpotri(L)
        if info:
            raise ValueError("Matrix is not symmetric-positive-definite")
        else:
            invM = np.triu(invM) + np.triu(invM, k=1).T
    # Otherwise we use the generic (and also slower) method that will
    # work if, due to numerical issues, the matrix is not quite SPD
    else:
        invM = np.linalg.inv(M)

    return invM


def camel_case_split_and_lowercase(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])'
                          '|(?<=[A-Z])(?=[A-Z][a-z])|$)',
                          identifier)
    return [m.group(0).lower() for m in matches]
