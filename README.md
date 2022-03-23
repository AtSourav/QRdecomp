# QRdecomp
UU course on Python and scientific programming: project

An overly elaborate and mildly humorous package for the QR decomposition of a real matrix.
Built using numpy. Numpy already has a function np.linalg.qr() to do this :/ 

Usage instructions:

from qrdecomposition_sourav import QRdecomposition as qrd  # to import the QRdecomposition class

Class methods are Qmatrix(), Rmatrix(), QR() to return the tuple of the two, FloatingPointErrorOrder()
Use help(...) to view the docstrings.

The input can be a numpy array or a list of ints or floats, output arrays are always numpy arrays of floats.

Can perform operations on two dimensional matrices, but you can try feeding in higher dimensional matrices.

A consistency check would be to feed in an already upper triangular matrix ;)  


## LICENSE
[ MIT ] (LICENSE)  
