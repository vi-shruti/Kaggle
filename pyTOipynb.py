# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:14:45 2019

@author: Shruti
"""

import IPython.nbformat.current as nbf
nb = nbf.read(open('revenueAnalysisUnDef.py', 'r'), 'py')
nbf.write(nb, open('revenueAnalysisUnDef.ipynb', 'w'), 'ipynb')