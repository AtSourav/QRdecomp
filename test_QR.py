 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sourav
"""

"""
Some tests for the package qrdecomposition_sourav.
"""

import numpy as np

from qrdecomposition_sourav import QRdecomposition as qrs

def createInstComplete(x):
    return qrs(x)

def createInstReduced(x):
    return qrs(x,'reduced')

def createInstIncorrect(x):
    return qrs(x,'not a mode')

def computeQR(x):
    return qrs(x).QR()

def computeQ(x):
    return qrs(x).Qmatrix()

def computeR(x):
    return qrs(x).Rmatrix()

def computeErr(x):
    return qrs(x).FloatingPointErrorOrder()

rtol_val = 1e-8
atol_val = 1e-12  # taking an absolute tolerance is important in cases where 
                  # there are elements that are exactly zero



# now the tests

M1 = np.random.rand(3,3)

def test_exception_mode(capfd):
    createInstIncorrect(M1)
    out, err = capfd.readouterr()
    assert out == 'The mode is unrecognized, please choose a valid mode.\n\n'
    
    

M2 = np.random.rand(3,3,3,2)
    
def test_exception_dimension(capfd):
    createInstComplete(M2)
    out, err = capfd.readouterr()
    assert out == 'Sorry, we can only work with a two dimensional matrix!\n\n'
    
     
    
M3 = np.random.rand(4,4)

def test_call_QR_first(capfd):
    inst = createInstReduced(M3)
    inst.FloatingPointErrorOrder()
    out, err = capfd.readouterr()
    assert out == 'You need to call the method QR() first.\n\n'
    
    
    
M4 = np.triu(np.random.rand(5,5))

def test_already_uppertriangular_complete(capfd):
    inst = createInstComplete(M4)
    inst.QR()
    out, err = capfd.readouterr()
    assert out == 'Dummy! The matrix is already upper triangular.\n\n'
    
def test_already_uppertriangular_reduced(capfd):
    inst = createInstReduced(M4)
    inst.Rmatrix()
    out, err = capfd.readouterr()
    assert out == 'Dummy! The matrix is already upper triangular.\n\n'
    
    
    
M5 = np.random.rand(6,6)

def test_QR_consistency_with_input():
    qr = createInstComplete(M5).QR()
    assert np.allclose(qr[0]@qr[1], M5, rtol=rtol_val,atol=atol_val)
    
def test_QR_consistency_Qmatrix():
    qr = createInstComplete(M5).QR()
    q = createInstComplete(M5).Qmatrix()
    assert np.all(q == qr[0])
    
def test_QR_consistency_Rmatrix():
    qr = createInstComplete(M5).QR()
    r = createInstComplete(M5).Rmatrix()
    assert np.all(r == qr[1])
    
def test_QR_uppertriangular():
    r = createInstComplete(M5).Rmatrix()
    assert np.all(np.triu(r) == r)
    
def test_QR_Qmatrix_orthogonality():
    q = createInstComplete(M5).Qmatrix()
    assert np.allclose(q.transpose()@q, np.eye(M5.shape[0]), rtol=1., atol=atol_val)
    
def test_QR_equivalence_modes_square_matrix():
    q_comp = createInstComplete(M5).QR()
    q_red = createInstReduced(M5).QR()
    assert np.all(q_comp[0] == q_red[0]) and np.all(q_comp[1] == q_red[1]) 
    
    
    
M5 = np.random.randint(100,size=(10,10))

def test_QR_consistency_with_input_int():
    qr = createInstComplete(M5).QR()
    assert np.allclose(M5,qr[0]@qr[1], rtol=rtol_val,atol=atol_val)
    
def test_QR_consistency_Qmatrix_int():
    qr = createInstComplete(M5).QR()
    q = createInstComplete(M5).Qmatrix()
    assert np.all(q == qr[0])
    
def test_QR_consistency_Rmatrix_int():
    qr = createInstComplete(M5).QR()
    r = createInstComplete(M5).Rmatrix()
    assert np.all(r == qr[1])
    
def test_QR_uppertriangular_int():
    r = createInstComplete(M5).Rmatrix()
    assert np.all(np.triu(r) == r)
    
def test_QR_Qmatrix_orthogonality_int():
    q = createInstComplete(M5).Qmatrix()
    assert np.allclose(q.transpose()@q, np.eye(M5.shape[0]), rtol=1., atol=atol_val)
    
def test_QR_equivalence_modes_square_matrix_int():
    q_comp = createInstComplete(M5).QR()
    q_red = createInstReduced(M5).QR()
    assert np.all(q_comp[0] == q_red[0]) and np.all(q_comp[1] == q_red[1]) 
    
    

M6 = np.random.rand(3,5)
    
def test_QR_consistency_with_input_more_columns():
    qr = createInstComplete(M6).QR()
    assert np.allclose(qr[0]@qr[1], M6, rtol=rtol_val,atol=atol_val)
    
def test_QR_uppertriangular_more_columns():
    r = createInstComplete(M6).Rmatrix()
    assert np.all(np.triu(r) == r)
    
def test_QR_Qmatrix_orthogonality_more_columns():
    q = createInstComplete(M6).Qmatrix()
    assert np.allclose(q.transpose()@q, np.eye(M6.shape[0]), rtol=1., atol=atol_val)
    
def test_QR_equivalence_modes_more_columns():
    q_comp = createInstComplete(M6).QR()
    q_red = createInstReduced(M6).QR()
    assert np.all(q_comp[0] == q_red[0]) and np.all(q_comp[1] == q_red[1])
    
    
    
M6 = np.random.randint(50,size=(12,16))
    
def test_QR_consistency_with_input_more_columns_int():
    qr = createInstComplete(M6).QR()
    assert np.allclose(qr[0]@qr[1], M6, rtol=rtol_val,atol=atol_val)
    
def test_QR_uppertriangular_more_columns_int():
    r = createInstComplete(M6).Rmatrix()
    assert np.all(np.triu(r) == r)
    
def test_QR_Qmatrix_orthogonality_more_columns_int():
    q = createInstComplete(M6).Qmatrix()
    assert np.allclose(q.transpose()@q, np.eye(M6.shape[0]), rtol=1., atol=atol_val)
    
def test_QR_equivalence_modes_more_columns_int():
    q_comp = createInstComplete(M6).QR()
    q_red = createInstReduced(M6).QR()
    assert np.all(q_comp[0] == q_red[0]) and np.all(q_comp[1] == q_red[1])
    
  
    
M7 = np.random.rand(5,3)

def test_QR_consistency_with_input_more_rows_complete():
    qr = createInstComplete(M7).QR()
    assert np.allclose(qr[0]@qr[1], M7, rtol=rtol_val,atol=atol_val)
    
def test_QR_consistency_with_input_more_rows_reduced():
    qr = createInstReduced(M7).QR()
    assert np.allclose(qr[0]@qr[1], M7, rtol=rtol_val,atol=atol_val)
    
def test_QR_uppertriangular_more_rows_complete():
    r = createInstComplete(M7).Rmatrix()
    assert np.all(np.triu(r) == r)
    
def test_QR_uppertriangular_more_rows_reduced():
    r = createInstReduced(M7).Rmatrix()
    assert np.all(np.triu(r) == r)
    
def test_QR_Qmatrix_orthogonality_more_rows_complete():
    q = createInstComplete(M7).Qmatrix()
    assert np.allclose(q.transpose()@q, np.eye(M7.shape[0]), rtol=1., atol=atol_val)
    
def test_QR_Qmatrix_orthogonality_more_rows_reduced():
    q = createInstReduced(M7).Qmatrix()
    assert np.allclose(q.transpose()@q, np.eye(M7.shape[1]), rtol=1., atol=atol_val)
    
def test_QR_Iequality_modes_more_rows():
    q_comp = createInstComplete(M7).QR()
    q_red = createInstReduced(M7).QR()
    assert not ((q_comp[0].shape == q_red[0].shape) or (q_comp[1].shape == q_red[1].shape))
    
def test_QR_equivalence_modes_more_rows():
    q_comp = createInstComplete(M7).QR()
    q_red = createInstReduced(M7).QR()
    assert np.all(q_comp[0][:,:M7.shape[1]] == q_red[0]) and np.all(q_comp[1][:M7.shape[1],:] == q_red[1])
    
    
    
M7 = np.random.randint(1000,size=(50,35))

def test_QR_consistency_with_input_more_rows_complete_int():
    qr = createInstComplete(M7).QR()
    assert np.allclose(qr[0]@qr[1], M7, rtol=rtol_val,atol=atol_val)
    
def test_QR_consistency_with_input_more_rows_reduced_int():
    qr = createInstReduced(M7).QR()
    assert np.allclose(qr[0]@qr[1], M7, rtol=rtol_val,atol=atol_val)
    
def test_QR_uppertriangular_more_rows_complete_int():
    r = createInstComplete(M7).Rmatrix()
    assert np.all(np.triu(r) == r)
    
def test_QR_uppertriangular_more_rows_reduced_int():
    r = createInstReduced(M7).Rmatrix()
    assert np.all(np.triu(r) == r)
    
def test_QR_Qmatrix_orthogonality_more_rows_complete_int():
    q = createInstComplete(M7).Qmatrix()
    assert np.allclose(q.transpose()@q, np.eye(M7.shape[0]), rtol=1., atol=atol_val)
    
def test_QR_Qmatrix_orthogonality_more_rows_reduced_int():
    q = createInstReduced(M7).Qmatrix()
    assert np.allclose(q.transpose()@q, np.eye(M7.shape[1]), rtol=1., atol=atol_val)
    
def test_QR_Iequality_modes_more_rows_int():
    q_comp = createInstComplete(M7).QR()
    q_red = createInstReduced(M7).QR()
    assert not ((q_comp[0].shape == q_red[0].shape) or (q_comp[1].shape == q_red[1].shape))
    
def test_QR_equivalence_modes_more_rows_int():
    q_comp = createInstComplete(M7).QR()
    q_red = createInstReduced(M7).QR()
    assert np.all(q_comp[0][:,:M7.shape[1]] == q_red[0]) and np.all(q_comp[1][:M7.shape[1],:] == q_red[1])
    
    
    

    
    
    
    
    
