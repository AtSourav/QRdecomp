  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sourav
"""

     
import numpy as np
import copy

from . import CustomExceptions
from .Householder import Householder
# '.' refers to the fact that we are importing the respecting modules from the current package. Otherwise when importing the QRdecomp package (which means we are outside the package) which basically imports the __init__ module of the package, the path from which the module should be imported is not specified.


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
Now we define a function to calculate the order of a (positive) floating point number in base 10. 
Take a floating point number n and express it in scientific notation ie <sign>x.yz...e<sign><abs exponent>, then the order of the number is 10^<sign><exponent>, we want to calculate <sign><abs exponent>
"""

def order10(flo):
    return int(np.floor(np.log10(flo)))



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
                
            arrayQ = np.array(matrix,dtype='float64')   
            
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
                self.__R = np.triu(R)
                
                # we create some private instance variables to store the calculated matrices 
                # these attributes can't be called before the method QR() is called, as they don't exist until the method QR() is applied
               
            elif self.__mode=='reduced':
                self.__Q = Q[:,:c]
                self.__R = np.triu(R[:c,:])
                
                # we are using np.triu to convert the lower triangular elements to zero by hand
                # the code itself produces very small numbers (compared to the non-zero elements of the R matrix) instead of exact zeroes because of inaccuracies in floating point arithmetic
                # we have checked the code without using the np.triu
                # now we're using it for aesthetic reasons, to put the floating point errors in the lower triangular part of R exactly to zero
                # np.linalg.qr() does the same, check the source code, although I'm not sure if they perform some additional checks on the output, they use a FORTRAN linear algebra library, I can't tell what the routines from that library are really doing 
                
            self.__fperror = order10(np.amax(np.abs(np.tril(R,-1)))) 
            # this attribute gives the order (base10) to which we are ignoring the floating point error when we set the very small lower triangular elements of R to zero, it doesn't matter what the mode is
            # in other words, it says that R is upper triangular upto an error of x.yz...e<self.__fperror>
             
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
        
    def FloatingPointErrorOrder(self):
        return 'The floating point error is at best of the order of 10^%d' %self.__fperror
            
            
            
            
            
            
            
            
            
            
            