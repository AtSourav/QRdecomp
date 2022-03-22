  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sourav
"""

"""Module for the Householder transform for the first column of a given matrix"""



import numpy as np

s=-1        # s=-1 chosen so the Householder transform of [x,0,0,...] is itself.


#%%

def mod_vec_signed(v):
    """
    Norm with a possible additional sign.
    
    Parameters
    ----------
    object: numpy.ndarray
        A one dimensional array (of floats).
        
    Returns
    -------
    out : float
        The Euclidean (Frobenius) norm of the input array times the sign of the
        first element.
        
    Examples
    --------
    >>> from QRdecomp.Householder import mod_vec_signed
    >>> import numpy as np
    >>> a = np.ones(4)
    >>> mod_vec_signed(a)
    2.0
    >>> b = np.array([-1,1,4,5])
    >>> mod_vec_signed(b)
    -6.557438524302   
          
        
        
    """
    
    return np.sign(v[0])*np.linalg.norm(v)


#%%

def find_u(matrix, size):
    """
    Returns the u vector when applying the Householder transform to the first
    column of a matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        A two dimensional array of floats.
    size : int
        The number of rows in the input matrix. For our purposes this will 
        already be known so we can use this as one of the inputs.

    Returns
    -------
    u_res : numpy.ndarray
        It will return a one dimensional array of floats as follows.
        Say M is the input matrix. Let v be the first column vector in M. 
        Let v = (x1,x2,x3,...). This function should return the vector 
        u = (x1 + s*sign(x1)*||v||, x2, x3, ...)
        
    Examples
    --------
    >>> import numpy as np
    >>> from QRdecomp.Householder import find_u
    >>> M = np.array([[2.,-2.,18.],[2.,1.,0],[1.,2.,0]])
    >>> find_u(M, 3)
    array([-1.,  2.,  1.])
        
        
    """
        
    u = matrix[:,0]
    u_res = u + s*mod_vec_signed(u)*np.hstack(([1],np.zeros(size-1)))
    
    return u_res

    
#%%

def Householder(matrix,size):
    """
    Returns the Householder matrix to transform the first column of the input 
    matrix to the form (x,0,0,...)

    Parameters
    ----------
    matrix : numpy.ndarray
        A two dimensional array of floats.
    size : int
        The number of rows in the input matrix. For our purposes this will 
        already be known so we can use this as one of the inputs.

    Returns
    -------
    out: numpy.ndarray
        Will return a Householder transform to rotate the first column of the
        input matrix to the first axis. Consider M= ([c1],[c2],[c3],...)
        where c1,c2,c3,... are the columns of M. This functions returns the
        matrix H, such that H.c1 = -s||c1|| e1, where e1 is the unit vector
        (1,0,0,0,...). H is a square matrix of dimension equal to the number
        of rows in M.
        
    Examples
    --------
    >>> import numpy as np
    >>> from QRdecomp.Householder import Householder
    >>> M = np.random.rand(3,3)
    >>> M
    array([[0.53225326, 0.16145292, 0.58382   ],
           [0.97964261, 0.87404203, 0.78714679],
           [0.8684117 , 0.81036399, 0.0806642 ]])
    >>> Householder(M, 3)
    array([[ 0.37663003,  0.69320913,  0.61450055],
           [ 0.69320913,  0.22912729, -0.68334602],
           [ 0.61450055, -0.68334602,  0.39424268]])
    >>> Householder(M,3)@M
    array([[ 1.41319922e+00,  1.16467105e+00,  8.15109679e-01],
           [ 1.31626219e-16, -2.41571483e-01,  5.29944606e-01],
           [-2.14270285e-17, -1.78580165e-01, -1.47334649e-01]])
        
    """
    
    inner = np.inner(find_u(matrix, size),find_u(matrix, size))
    subtract = 0.
    if inner != 0:
        subtract = 2*np.outer(find_u(matrix, size),find_u(matrix, size))/inner
        
    # the case of inner=0 corresponds to when the Householder transform should 
    # the identity, for example when matrix is already upper triangular
    
    return np.eye(size) - subtract

   