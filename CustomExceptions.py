#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sourav
"""


class DimensionError(Exception):
    """to throw an exception for dimensions other than two"""
    pass

class ModeUnrecognized(Exception):
    """to throw when the mode is not recognized"""
    pass

class Pointless(Exception):
    """to throw when the input matrix is already upper triangular"""
    pass

class CallQR(Exception):
    """to throw when the method QR() needs to be called first"""
    pass


