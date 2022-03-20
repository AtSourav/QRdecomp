  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sourav
"""

     
import numpy as np
import copy

import CustomExceptions
from Householder import Householder


#%%

"""First, a quick command to pad a matrix from the top left with ones on the diagonal and zeroes everywhere else"""


def pad(mat,step):
    
    #step is the number of layers of padding we want
    #mat is a square matrix of floats (or ints), the output will be in floats
    
    size = mat.shape[0]
    
    mat = np.pad(mat, [(step,0),(step,0)],mode='constant',constant_values=0)
    
    additive = np.pad(np.eye(step),[(0,size),(0,size)],mode='constant',constant_values=0)
    
    return mat + additive


#%%

"""
Now onto the main class
"""



#the input matrix need not be a numpy array
#the input can be in floats or ints
#we'll get the output as the tuple Q,R and also as individual methods Q, and R
#we'll keep different modes for when the matrix is not square (we have defined another exception regarding this matter)

class QRdecomposition:
    
    def __init__(self, matrix, mode='complete'):
        try:
            if mode not in ['complete','reduced']:
                raise CustomExceptions.ModeUnrecognized
                
            arrayQ = np.array(matrix,dtype=float)   
            
            #arrayQ is a local variable within the __init__ fn and cannot be accessed from anywhere else. Note that we can also create public or private instance variables.
            
            if arrayQ.ndim == 2:
                self.__array = arrayQ
                self.__mode = mode
                #self.__array creates a (private) instance variable named _QRdecomposition__array
                #it's not possible to access this variable from outside the class unless using the mangled up name.
            else:
                raise CustomExceptions.DimensionError
                
        except CustomExceptions.ModeUnrecognized:
            print('The mode is unrecognized, please choose a valid mode.')
            print()
            
        except CustomExceptions.DimensionError:
            print('Sorry, we can only work with a two dimensional matrix!')
            print()
            
            
            
            
    def QR(self):
        #this should return the tuple Q,R
        
        if '_QRdecomposition__Q' in dir(self):
            return self.__Q, self.__R
            # saves us computation if this method has been called already for the given instance
        else:
            R = copy.deepcopy(self.__array)   
            # we made a deepcopy of self.__array as we don't want it to change as we change R, otherwise calling the QR() method more than once will give incorrect (and new) results everytime as it will decomposing a new matrix everytime
            # we'll keep saving R in place as we turn it upper triangular
            
            step = 0
            size = min(R.shape)
            r = R.shape[0]
            c = R.shape[1]
            Q = np.eye(r)   #The complete H matrix is (r,r)
                            #we'll keep saving this in place as we multiply with Householder transforms
            
            for step in range(size):
                Rredu = R[step:,step:,]
                Hredu = Householder(Rredu,r-step)
                
                np.matmul(Hredu,Rredu, out=Rredu)
                # the out=<...> specifies the location of the output, so we have ensured an in-place multiplication
                # Rredu is a view of R, so we have automatically updated the relevant block in R
                
                H = pad(Hredu,step)
                np.matmul(Q,H , out=Q)
                # we are multiplying Q from the right with Hredu padded up to shape r,r as H, and then storing in place
                
                step +=1
                
            if self.__mode=='complete':
                self.__Q = Q
                self.__R = R
                # we create some private instance variables to store the calculated matrices 
                # these attributes can't be called before the method QR() is called, as they don't exist until the method QR() is applied
               
            elif self.__mode=='reduced':
                self.__Q = Q[:,:c]
                self.__R = R[:c,:]
                
            
             
            return self.__Q, self.__R  #convert into nice looking results? put numerical zeros to actual zeros?
                
            
        
        
    def Qmatrix(self):
        if '_QRdecomposition__Q' in dir(self):
            return self.__Q
        else:
            self.QR()
            return self.__Q
        
        # dir(instance) gives a list of methods and attributes in <instance> 
        # this saves us some computation if Q has been calculated once already
        
        
    def Rmatrix(self):
        if '_QRdecomposition__R' in dir(self):
            return self.__R
        else:
            self.QR()
            return self.__R
            
            
            
            
            
            
            
            
            
            
            