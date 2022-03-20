  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sourav
"""

"""We intend to make a module to calculate the Householder transform 
for the first column of any given matrix"""



import numpy as np

s=-1
# we choose s=-1 so eventually if we use am upper triangular matrix as an input,
# QRdecomposition gives it back (and the Q matrix is identity)
# choosing s=1 puts an overall minus sign
# is one choice faster than the other?

#%%
def mod_vec_signed(v):
    # returns the norm given an array
    
    return np.sign(v[0])*np.linalg.norm(v)


#%%

def find_u(matrix, size):
    
    # the size of the matrix (number of rows) will be given as it will be known
    # already and we won't have to calculate it again
    # the matrix is to be input as an array
    # it should be an array of floats
    
    u = matrix[:,0]
    u_res = u + s*mod_vec_signed(u)*np.hstack(([1],np.zeros(size-1)))
    #we shouldn't do an in place operation on u after slicing from matric, as it
    #will change matrix
    
    return u_res

    
#%%

def Householder(matrix,size):
    
    # should return the Householder matrix H
    inner = np.inner(find_u(matrix, size),find_u(matrix, size))
    subtract = 0.
    if inner != 0:
        subtract = 2*np.outer(find_u(matrix, size),find_u(matrix, size))/inner
        
    # the case of inner=0 corresponds to when the Householder transform should 
    # the identity, for example when matrix is already upper triangular
    
    return np.eye(size) - subtract

    # np.outer(a,b) calculates a b^T where a,b are column vectors