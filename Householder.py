#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 19:23:42 2022

@author: sourav
"""

"""We intend to make a module to calculate the Householder transform 
for the first column of any given matrix"""

"""
I'm not yet sure how to deal with numerics tbh, I'm only making some changes
to try and fix some confusion on the git branches
"""

import numpy as np

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
    u += mod_vec_signed(u)*np.hstack(([1],np.zeros(size-1)))
    
    return u

    
#%%

def Householder(matrix,size):
    
    # should return the Householder matrix H
    inner = np.inner(find_u(matrix, size),find_u(matrix, size))
    subtract = 2*np.outer(find_u(matrix, size),find_u(matrix, size))/inner
    
    return np.eye(size) - subtract

    # np.outer(a,b) calculates a b^T where a,b are column vectors