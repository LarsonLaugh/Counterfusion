# Copyright 2024 Lixian Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import warnings, copy, random, json

__all__ = ['pmatrix','qmatrix','lmatrix','replace_colnum',
           'det','theta','merge','interaction_builder'
           ]


def pmatrix(Om1,Om2,nfm):
    m = Om1.shape[0]
    p = np.zeros((m-nfm,m-nfm))
    for k in range(m-nfm):
        for j in range(m-nfm):
            if j == k:
                p[k,k] = 1-np.dot(Om2[k+nfm,:nfm],Om1[:nfm,k+nfm])
            else:
                p[k,j] = -np.dot(Om2[k+nfm,:nfm],Om1[:nfm,j+nfm])
    return p

def qmatrix(Om1,Om2,nfm):
    m = Om1.shape[0]
    q = np.zeros((m-nfm,nfm))
    for k in range(m-nfm):
        for j in range(nfm):
            q[k,j] = np.dot(Om2[k+nfm,:nfm],Om1[:nfm,j])
    return q

def lmatrix(Om,nfm):
    Om_copy = copy.deepcopy(Om)
    return Om_copy[nfm:,nfm:]

def replace_colnum(matrix,col_num,col_replace):
    matrix_copy = copy.deepcopy(matrix)
    matrix_copy[:,col_num]=col_replace
    return matrix_copy

def det(matrix):
    return np.linalg.det(matrix)

def theta(Om1,Om2,nfm):
    m = Om1.shape[0]
    th = np.zeros((m,m))
    p = pmatrix(Om1, Om2, nfm)
    q = qmatrix(Om1, Om2, nfm)
    l = lmatrix(Om2, nfm)
    detp = det(p)
    for k in range(m):
        for i in range(m):
            if k<nfm and i<nfm:
                th[k,i] = Om1[k,i]+sum([Om1[k,j]*det(replace_colnum(p,j-nfm,q[:,i]))/detp for j in range(nfm,m)])
            elif k<nfm and i>=nfm:
                th[k,i] = sum([Om1[k,j]*det(replace_colnum(p,j-nfm,l[:,i-nfm]))/detp for j in range(nfm,m)])
            elif k>=nfm and i<nfm:
                th[k,i] = det(replace_colnum(p,k-nfm,q[:,i]))/detp
            else:
                th[k,i] = det(replace_colnum(p,k-nfm,l[:,i-nfm]))/detp
    return th

def merge(Om1,Om2,nfm):
    m = Om1.shape[0]
    th = theta(Om1,Om2,nfm)
    Om3 = np.zeros((m,m))
    for k in range(m):
        for i in range(m):
            if k<nfm and i<nfm:
                Om3[k,i] = sum([Om2[k,j]*th[j,i] for j in range(nfm)])
            elif k<nfm and i>=nfm:
                Om3[k,i] = sum([Om2[k,j]*th[j,i] for j in range(nfm)])+Om2[k,i]
            elif k>=nfm and i<nfm:
                Om3[k,i] = sum([Om1[k,j]*th[j,i] for j in range(nfm,m)])+Om1[k,i]
            else:
                Om3[k,i] = sum([Om1[k,j]*th[j,i] for j in range(nfm,m)])
    return Om3

def interaction_builder(dimension,id1,id2,value):
    if isinstance(value,(int,float)):
        # Build a doubly stochastic matrix
        id1,id2 = int(id1),int(id2)
        matrix = np.eye(dimension)
        matrix[id1,id1] = 1-value/2
        matrix[id1,id2] = value/2
        matrix[id2,id1] = value/2
        matrix[id2,id2] = 1-value/2
    elif isinstance(value,list) and len(value)==2 and all(isinstance(v,(int,float)) for v in value):
        # Build a left stochastic matrix for Markov-process (each column summing to 1)
        id1,id2 = int(id1),int(id2)
        matrix = np.eye(dimension)
        matrix[id1,id1] = 1-value[0]
        matrix[id1,id2] = value[1]
        matrix[id2,id1] = value[0]
        matrix[id2,id2] = 1-value[1]
    else:
        raise ValueError("value needs to be either a float/int (doubly stochastic matrix) or a list including 2 "
                         "probabilities (left stochastic matrix) ")
    return matrix