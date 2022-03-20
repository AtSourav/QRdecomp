  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sourav
"""

     
import numpy as np
import CustomExceptions
from Householder import Householder


"""First, a quick command to pad a matrix from the top left with ones on the diagonal
     and zeroes everywhere else"""


def pad(mat,step):
    
    #step is the number of layers of padding we want
    #mat is a square matrix of floats (or ints), the output will be in floats
    
    size = mat.shape[0]
    
    mat = np.pad(mat, [(step,0),(step,0)],mode='constant',constant_values=0)
    
    additive = np.pad(np.eye(step),[(0,size),(0,size)],mode='constant',constant_values=0)
    
    return mat + additive




#the input matrix need not be a numpy array
#the input can be in floats or ints
#we'll get the output as the tuple Q,R and also as individual methods Q, and R
#we'll keep different modes for when the matrix is not square (we have defined 
# another exception regarding this matter)

class QRdecomposition:
    def __init__(self, matrix, mode='complete'):
        try:
            if mode not in ['complete','reduced']:
                raise CustomExceptions.ModeUnrecognized
                
            arrayQ = np.array(matrix,dtype=float)   
            
            #arrayQ is a local variable within the __init__ fn and cannot be accessed
            # from anywhere else. Note that we can also create public or private 
            # instance variables.
            
            if arrayQ.ndim == 2:
                self.array = arrayQ
            else:
                raise CustomExceptions.DimensionError
                
        except CustomExceptions.ModeUnrecognized:
            print('The mode is unrecognized, please choose a valid mode.')
            print()
            
        except CustomExceptions.DimensionError:
            print('Sorry, we can only work with a two dimensional matrix!')
            print()
            
            
            
            
            
            
            
            
            
            
            