  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sourav
"""

     
import numpy as np
import copy

from . import CustomExceptions
from .Householder import Householder


#%%

def pad(mat,step):
    """
    Pads a matrix from the top left with ones on the diagonal and zeroes 
    everywhere else.

    Parameters
    ----------
    mat : numpy.ndarray
        A two dimensional array of floats (or ints).
    step : int
        A positive integer denoting the number of layers of padding.

    Returns
    -------
    out: numpy.ndarray
        A array of floats (even if the input mat was in int), where mat 
        has been padded by step number of layers from the top left, with ones 
        on the diagonal and zero everywhere else.
        
    Examples
    --------
    >>> import numpy as np
    >>> from QRdecomp.main import pad
    >>> A = np.random.randint(10, size=(3,3))
    >>> A
    array([[8, 0, 3],
           [4, 5, 3],
           [7, 6, 6]])
    >>> pad(A,1)
    array([[1., 0., 0., 0.],
           [0., 8., 0., 3.],
           [0., 4., 5., 3.],
           [0., 7., 6., 6.]])
    >>> B = np.random.rand(2,2)
    array([[0.66249402, 0.54029319],
           [0.5604081 , 0.45140005]])
    >>> pad(B,3)
    array([[1.        , 0.        , 0.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        , 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.66249402, 0.54029319],
           [0.        , 0.        , 0.        , 0.5604081 , 0.45140005]])
    

    """
   
    
    size = mat.shape[0]
    
    mat = np.pad(mat, [(step,0),(step,0)],mode='constant',constant_values=0)
    
    additive = np.pad(np.eye(step),[(0,size),(0,size)],mode='constant',constant_values=0)
    
    return mat + additive


#%%

def order10(flo):
    """
    Calculates the order of a (positive) floating point number in base 10.

    Parameters
    ----------
    flo : float
        A positive floating point number.

    Returns
    -------
    out: int
        A positive or negative integer denoting the order of the input float in
        base 10. Express the floating point number flo in scientific notation 
        ie <+>x.yz...e<sign><abs exponent>, then the order of the number is 
        10^<sign><abs exponent>. This function outputs <sign><abs exponent>.
        
    Examples
    --------
    >>> import numpy as np
    >>> from QRdecomp.main import order10
    >>> order10(0.1234)
    -1
    >>> order10(np.sqrt(2599))
    1
    >>> order10(0.000000000235)
    -10
    """

    return int(np.floor(np.log10(flo)))



#%%

class QRdecomposition:
    """
    Instantiates a class for the QR decomposition of a two dimensional array.
    
    Parameters
    ----------
    matrix : array_like
        An array (a numpy.ndarray or a list or any other object that's converted 
        to a numpy.ndarray by numpy.array) of integers or floats. 
    mode : {'complete','reduced'} optional
        If mode='complete' (default), a complete QR decomposition is obtained.
        If mode='reduced', we obtain a reduced QR decomposition which is distinct
        from the complete QR decomposition when the number of rows > number of 
        columns in the input matrix.
        
    Raises
    ------
    'The mode is unrecognized, please choose a valid mode.'
        If mode not in {'complete','reduced'}.
        
    'Sorry, we can only work with a two dimensional matrix!'   
        If the input matrix is not two dimensional.
        
    Returns
    -------
    out: class qrdecomposition_sourav.main.QRdecomposition
        The output is a QRdecomposition
        class instance with some (private) attributes and methods relevant to the 
        QR decomposition of the input matrix.
        
    See Also
    --------
    numpy.linalg.qr: numpy provided inbuilt function for QR decomposition.
        
    Examples
    --------
    >>> import numpy as np
    >>> from QRdecomp import QRdecomposition
    >>> A=np.array([[2,-2,18],[2,1,0],[1,2,0]])
    >>> QRA = QRdecomposition(A)
    >>> type(QRA)
    __main__.QRdecomposition
    >>> B = [[2,3,7],[1,3,2],[7,9,0]]
    >>> QRB = QRdecomposition(B,'reduced')
    >>> type(QRB)
    __main__.QRdecomposition
    >>>QRB2 = QRdecomposition(B,'not in list')
    The mode is unrecognized, please choose a valid mode.
    >>> C = np.random.rand(4,3)
    >>> C
    array([[0.94736141, 0.33226513, 0.87538724],
           [0.17070941, 0.37403276, 0.75319867],
           [0.37496236, 0.10273167, 0.62695294],
           [0.01891008, 0.50151402, 0.96066435]])
    >>> QRC = QRdecomposition(C)
    >>> type(QRC)
    __main__.QRdecomposition
    >>> D = np.random.rand(3,2,4)
    >>> D
    array([[[0.82479318, 0.43394209, 0.43307473, 0.92062568],
            [0.92035304, 0.82966978, 0.6720225 , 0.82205618]],

           [[0.6071503 , 0.17213224, 0.11951942, 0.8517612 ],
            [0.45166216, 0.09570765, 0.90400416, 0.55215103]],

           [[0.39489499, 0.7689566 , 0.45105273, 0.73672948],
            [0.77310856, 0.32992669, 0.48690335, 0.24372514]]])
    >>> QRD = QRdecomposition(D)
    Sorry, we can only work with a two dimensional matrix!
    """
    
    def __init__(self, matrix, mode='complete'):
        try:
            if mode not in ['complete','reduced']:
                raise CustomExceptions.ModeUnrecognized
                
            arrayQ = np.array(matrix,dtype='float64')   
            
            if arrayQ.ndim == 2:
                self.__array = arrayQ
                self.__mode = mode
            else:
                raise CustomExceptions.DimensionError
                
        except CustomExceptions.ModeUnrecognized:
            print('The mode is unrecognized, please choose a valid mode.')
            print()
            
        except CustomExceptions.DimensionError:
            print('Sorry, we can only work with a two dimensional matrix!')
            print()
            
            
            
            
    def QR(self):
        """
        A QRdecomposition class method to calculate the tuple Q, R for the input
        matrix.
        
        Raises
        ------
        'Dummy! The matrix is already upper triangular.'
            If the input matrix is already upper triangular.

        Returns
        -------
        Q : numpy.ndarray
            The orthonormal Q matrix in the QR decomposition of the input matrix.
            If the input is already upper triangular, Q should be the identity 
            (as per conventions chosen for s). However we raise an exception in
            this case as nothing needs to be computed.
            
            If mode='complete' (default), Q is a square matrix with dimension 
            equal to the number of rows in the input matrix.
            
            If mode='reduced', and the input matrix has dimensions r,c with r>c,
            then Q has dimensions r,c.
            
        R : numpy.ndarray
            The upper triangular R matrix in the QR decomposition of the input
            matrix. If the input is already upper triangular, the same should be
            returned as a numpy.ndarray of floats (we raise an exception, and
            avoid the case altogether, it's just used for choosing conventions'). 
            
            If mode='complete', R has the same dimensions as the input matrix.
            
            If mode='reduced', and r>c, then R is a square matrix with dimension c.
           
        See Also
        --------
        QRdecomp.Householder: module for the Householder transform used herein.
        QRdecomp.CustomExceptions: module for the custom exceptions raised.
        numpy.linalg.qr: numpy provided inbuilt function for QR decomposition.
            
        Examples
        --------
        >>> import numpy as np
        >>> from QRdecomp import QRdecomposition
        >>> M1=np.random.randint(5,size=(3,3))
        >>> M1
        array([[0, 4, 1],
               [4, 2, 0],
               [3, 2, 1]])
        >>> QRdecomposition(M1).QR()
        (array([[ 1.        ,  0.        ,  0.        ],
                [ 0.        , -0.70710678,  0.70710678],
                [ 0.        , -0.70710678, -0.70710678]]),
         array([[ 0.        ,  4.        ,  1.        ],
                [ 0.        , -2.82842712, -0.70710678],
                [ 0.        ,  0.        , -0.70710678]]))
        >>> M2=np.triu(np.random.rand(4,4))
        >>> M2
        array([[0.88852557, 0.26433006, 0.42867313, 0.15930436],
               [0.        , 0.79417449, 0.50949558, 0.65408098],
               [0.        , 0.        , 0.24425774, 0.38858496],
               [0.        , 0.        , 0.        , 0.54905233]])
        >>> inst2 = QRdecomposition(M2)
        >>> inst2.QR()
        Dummy! The matrix is already upper triangular.
        >>> M3 = np.random.randint(10,size=(5,3))
        >>> M3
        array([[6, 2, 8],
               [5, 2, 4],
               [8, 0, 3],
               [8, 9, 1],
               [6, 0, 6]])
        >>> inst3 = QRdecomposition(M3)
        >>> inst3.QR()
        (array([[ 0.4       ,  0.07184854,  0.72581846,  0.32919083,  0.4468319 ],
                [ 0.33333333,  0.01260501,  0.16432914, -0.92156162,  0.11158006],
                [ 0.53333333,  0.47394827, -0.60381223,  0.13169552,  0.33014491],
                [ 0.53333333, -0.80230871, -0.19781058,  0.13163795, -0.12409155],
                [ 0.4       ,  0.3554612 ,  0.20607101,  0.08766588, -0.81455311]]),
         array([[15.        ,  6.26666667,  9.06666667],
                [ 0.        , -7.0518713 ,  3.37751169],
                [ 0.        ,  0.        ,  5.69104299],
                [ 0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.        ,  0.        ]]))
        >>> inst3red = QRdecomposition(M3,'reduced')
        >>> inst3red.QR()
        (array([[ 0.4       ,  0.07184854,  0.72581846],
                [ 0.33333333,  0.01260501,  0.16432914],
                [ 0.53333333,  0.47394827, -0.60381223],
                [ 0.53333333, -0.80230871, -0.19781058],
                [ 0.4       ,  0.3554612 ,  0.20607101]]),
         array([[15.        ,  6.26666667,  9.06666667],
                [ 0.        , -7.0518713 ,  3.37751169],
                [ 0.        ,  0.        ,  5.69104299]]))

        """
        
        if '_QRdecomposition__Q' in dir(self):
            return self.__Q, self.__R                   # saves us computation if this method has been called already for the given instance
        else:
            R = copy.deepcopy(self.__array) 
            
            max_lower_triangle = np.amax(np.abs(np.tril(R,-1)))
            try:
                if max_lower_triangle == 0.:
                    raise CustomExceptions.Pointless
                else:
                    step = 0
                    size = min(R.shape)
                    r = R.shape[0]
                    c = R.shape[1]
                    Q = np.eye(r)   
                    
                    for step in range(size):
                        Rredu = R[step:,step:,]
                        Hredu = Householder(Rredu,r-step)
                        
                        np.matmul(Hredu,Rredu, out=Rredu)
                        
                        H = pad(Hredu,step)
                        np.matmul(Q,H , out=Q)
                                                
                        step +=1
                        
                    if self.__mode=='complete':
                        self.__Q = Q
                        self.__R = np.triu(R)
                        
                       
                    elif self.__mode=='reduced':
                        self.__Q = Q[:,:c]
                        self.__R = np.triu(R[:c,:])
                        
                    self.__Runchanged = R
                        
                                             
                    return self.__Q, self.__R  
                
            except CustomExceptions.Pointless:
                print('Dummy! The matrix is already upper triangular.')
                print()
            
            
            
                
    def Qmatrix(self):
        """
        A QRdecomposition class method to calculate Q from the input matrix.
        Calls the QR() method if it hasn't already been called.
        
        Raises
        ------
        'Dummy! The matrix is already upper triangular.'
            If the input matrix is already upper triangular.

        Returns
        -------
        Q : numpy.ndarray
            The orthonormal Q matrix in the QR decomposition of the input matrix.
            If the input is already upper triangular, Q should be the identity 
            (as per conventions chosen for s). However we raise an exception in
            this case as nothing needs to be computed.
            
            If mode='complete' (default), Q is a square matrix with dimension 
            equal to the number of rows in the input matrix.
            
            If mode='reduced', and the input matrix has dimensions r,c with r>c,
            then Q has dimensions r,c.           
        
        See Also
        --------
        numpy.linalg.qr: numpy provided inbuilt function for QR decomposition.
            
        Examples
        --------
        >>> import numpy as np
        >>> from QRdecomp import QRdecomposition
        >>> M1=np.random.randint(5,size=(3,3))
        >>> M1
        array([[3, 4, 3],
               [4, 3, 3],
               [2, 3, 2]])
        >>> QRdecomposition(M1).Qmatrix()
        array([[ 0.55708601,  0.52062513,  0.64699664],
               [ 0.74278135, -0.66079343, -0.10783277],
               [ 0.37139068,  0.54064917, -0.75482941]])
        >>> M2=np.triu(np.random.rand(4,4))
        >>> M2
        array([[0.28004935, 0.03446024, 0.57372709, 0.41737163],
               [0.        , 0.80320048, 0.0364231 , 0.96655091],
               [0.        , 0.        , 0.63195612, 0.65311413],
               [0.        , 0.        , 0.        , 0.02293199]])
        >>> inst2 = QRdecomposition(M2)
        >>> inst2.Qmatrix()
        Dummy! The matrix is already upper triangular.
        >>> M3 = np.random.randint(10,size=(5,3))
        >>> M3
        array([[4, 8, 4],
               [5, 7, 5],
               [1, 8, 5],
               [1, 4, 5],
               [2, 7, 1]])
        >>> inst3 = QRdecomposition(M3)
        >>> inst3.Qmatrix()
        array([[ 0.58345997, -0.01117707, -0.13190749, -0.68891393,  0.40920357],
               [ 0.72932496,  0.38002024,  0.19436236,  0.31975827, -0.4285413 ],
               [ 0.14586499, -0.79077741,  0.20197241, -0.210073  , -0.51814185],
               [ 0.14586499, -0.26545531,  0.68096803,  0.33861348,  0.5743521 ],
               [ 0.29172998, -0.3995801 , -0.66356114,  0.51416194,  0.224841  ]])
        >>> inst3red = QRdecomposition(M3,'reduced')
        >>> inst3red.Qmatrix()
        array([[ 0.58345997, -0.01117707, -0.13190749],
               [ 0.72932496,  0.38002024,  0.19436236],
               [ 0.14586499, -0.79077741,  0.20197241],
               [ 0.14586499, -0.26545531,  0.68096803],
               [ 0.29172998, -0.3995801 , -0.66356114]])

        """
        if '_QRdecomposition__Q' in dir(self):
            return self.__Q
        else:
            self.QR()
            if '_QRdecomposition__Q' in dir(self):
                return self.__Q
            
            
        
    def Rmatrix(self):
        """
        A QRdecomposition class method to calculate R from the input matrix.
        Calls the QR() method if it hasn't already been called.
        
        Raises
        ------
        'Dummy! The matrix is already upper triangular.'
            If the input matrix is already upper triangular.

        Returns
        -------
        R : numpy.ndarray
            The upper triangular R matrix in the QR decomposition of the input
            matrix. If the input is already upper triangular, the same should be
            returned as a numpy.ndarray of floats (we raise an exception, and
            avoid the case altogether, it's just used for choosing conventions'). 
            
            If mode='complete', R has the same dimensions as the input matrix.
            
            If mode='reduced', and r>c, then R is a square matrix with dimension c.          
                    
        See Also
        --------
        numpy.linalg.qr: numpy provided inbuilt function for QR decomposition.
            
        Examples
        --------
        >>> import numpy as np
        >>> from QRdecomp import QRdecomposition
        >>> M1=np.random.randint(5,size=(3,3))
        >>> M1
        array([[3, 4, 3],
               [4, 3, 3],
               [2, 3, 2]])
        >>> QRdecomposition(M1).Rmatrix()
        array([[5.38516481, 5.57086015, 4.64238345],
               [0.        , 1.72206772, 0.66079343],
               [0.        , 0.        , 0.10783277]])
        >>> M2=np.triu(np.random.rand(4,4))
        >>> M2
        array([[0.28004935, 0.03446024, 0.57372709, 0.41737163],
               [0.        , 0.80320048, 0.0364231 , 0.96655091],
               [0.        , 0.        , 0.63195612, 0.65311413],
               [0.        , 0.        , 0.        , 0.02293199]])
        >>> inst2 = QRdecomposition(M2)
        >>> inst2.Rmatrix()
        Dummy! The matrix is already upper triangular.
        >>> M3 = np.random.randint(10,size=(5,3))
        >>> M3
        array([[4, 8, 4],
               [5, 7, 5],
               [1, 8, 5],
               [1, 4, 5],
               [2, 7, 1]])
        >>> inst3 = QRdecomposition(M3)
        >>> inst3.Rmatrix()
        array([[ 6.8556546 , 13.56544421,  7.73084455],
               [ 0.        , -7.6143761 , -3.82535078],
               [ 0.        ,  0.        ,  4.19532287],
               [ 0.        ,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ]])
        >>> inst3red = QRdecomposition(M3,'reduced')
        >>> inst3red.Rmatrix()
        array([[ 6.8556546 , 13.56544421,  7.73084455],
               [ 0.        , -7.6143761 , -3.82535078],
               [ 0.        ,  0.        ,  4.19532287]])

        """
        if '_QRdecomposition__R' in dir(self):
            return self.__R
        else:
            self.QR()
            if '_QRdecomposition__R' in dir(self):
                return self.__R
            
        
    def FloatingPointErrorOrder(self):
        """
        Gives an estimate of the floating point error involved in the QR decomposition.

        Raises
        ------
        'You need to call the method QR() first.'
            If the QR decomposition of the input matrix hasn't been performed yet.

        Returns
        -------
        out : string 'The floating point error is at best of the order of 10^<error order>'
              or string 'There are no floating point errors in the lower triangular elements of R up to the max precision at 0.'  
              
              The <error> in the first case is the order to which we neglect the
              floating point error in producing an upper triangular R, ie the
              lower triangular 0. elements of R are accurate upto <sign>x. 10^<error order>,
              where x is single digit.
              
              The second case is self-explanatory.
              
        Examples
        --------
        >>> M3 = np.random.randint(10,size=(5,3))
        >>> M3
        array([[4, 8, 4],
               [5, 7, 5],
               [1, 8, 5],
               [1, 4, 5],
               [2, 7, 1]])
        >>> inst3 = QRdecomposition(M3)
        >>> inst3.FloatingPointErrorOrder()
        'The floating point error is at best of the order of 10^-16'            

        """
        try:
            if '_QRdecomposition__R' in dir(self):
                max_lower_triangle = np.amax(np.abs(np.tril(self.__Runchanged,-1)))
                if max_lower_triangle > 0.:
                    self.__fperror = order10(max_lower_triangle)    
                    return 'The floating point error is at best of the order of 10^%d' %self.__fperror
                else:
                    return 'There are no floating point errors in the lower triangular elements of R up to the max precision at 0.'
            else:
                raise CustomExceptions.CallQR
        
        except CustomExceptions.CallQR:
            print('You need to call the method QR() first.')
            print()
            
            
            
            
            
            
            
            
            
            
            