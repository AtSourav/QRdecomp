#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sourav
"""

"""
We define some custom exceptions to throw when the array/list is not 
two dimensional, and when the mode specified is not recognized.
These will inherit from the default Exception class in python.
"""

class DimensionError(Exception):
    #to throw an exception for dimensions other than two
    pass


class ModeUnrecognized(Exception):
    #to throw when the mode is not recognized
    pass


