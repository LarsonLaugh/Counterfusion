import numpy as np
import copy
import random


# TODO: add this class to create a system consisting of electron reserviors (contacts) and interactions in between
class system:
    def __init__(self,current,diagram):
        ''' This diagram should be a list containing effective_matrix instances between adjacent contacts
        along the forward-propagation direction.
        '''
        self.term_current = current
        self.diagram = diagram
        self.m = diagram[0].trans_mat().shape[0]
        self.num_term = len(diagram)
    def mastermat(self,mf):
        effective_matrices = self.diagram
        num_term = self.num_term
        mat = np.zeros((num_term,num_term))
        for t in range(num_term):
            premat, aftmat = effective_matrices[t - 1].trans_mat(), effective_matrices[t].trans_mat()
            mat[t, t] = self.m - premat[:mf, mf:].sum() - aftmat[mf:, :mf].sum()
            mat[t, t - 1] = -premat[:mf, :mf].sum()
            if t == num_term-1:
                mat[t, 0] = -aftmat[mf:, mf:].sum()
            else:
                mat[t, t + 1] = -aftmat[mf:, mf:].sum()
        return mat
    def solve(self,mf):
        term_voltage = np.linalg.solve(self.mastermat(mf),self.term_current)
        return term_voltage
class effective_matrix:
    def __init__(self,sequence,m,mf):
        self.seq = sequence
        self.m = m
        self.mf = mf
    def get_seq(self):
        return self.seq
    def trans_mat(self):
        seq = self.seq
        m = self.m
        mf = self.mf
        matrices = []
        # Iterate through each element (interaction type 't' and strength 'v') in 'seq'.
        for id1, id2, v in zip(seq[:, 0], seq[:, 1], seq[:, 2]):
            matrix = scatter_matrix(m, id1, id2, v)
            matrices.append(matrix)
        # Initialize the result matrix 'mat0' with the first matrix in the 'matrices' list.
        mat0 = matrices[0]
        # Merge the matrices in 'matrices' list sequentially and update 'mat0' with the result.
        for mat1 in matrices[1:]:
            res = merge(mat0, mat1, mf)
            mat0 = res
        # Return the final merged matrix 'res' and the input sequence 'seq'.
        return res

#================================================================
#core functions

def pmatrix(Om1,Om2,mf):
    m = Om1.shape[0]
    p = np.zeros((m-mf,m-mf))
    for k in range(m-mf):
        for j in range(m-mf):
            if j == k:
                p[k,k] = 1-np.dot(Om2[k+mf,:mf],Om1[:mf,k+mf])
            else:
                p[k,j] = -np.dot(Om2[k+mf,:mf],Om1[:mf,j+mf])
    return p

def qmatrix(Om1,Om2,mf):
    m = Om1.shape[0]
    q = np.zeros((m-mf,mf))
    for k in range(m-mf):
        for j in range(mf):
            q[k,j] = np.dot(Om2[k+mf,:mf],Om1[:mf,j])
    return q

def lmatrix(Om,mf):
    Om_copy = copy.deepcopy(Om)
    return Om_copy[mf:,mf:]

def replace_colnum(matrix,col_num,col_replace):
    matrix_copy = copy.deepcopy(matrix)
    matrix_copy[:,col_num]=col_replace
    return matrix_copy

def det(matrix):
    return np.linalg.det(matrix)

def theta(Om1,Om2,mf):
    m = Om1.shape[0]
    th = np.zeros((m,m))
    p = pmatrix(Om1, Om2, mf)
    q = qmatrix(Om1, Om2, mf)
    l = lmatrix(Om2, mf)
    detp = det(p)
    for k in range(m):
        for i in range(m):
            if k<mf and i<mf:
                th[k,i] = Om1[k,i]+sum([Om1[k,j]*det(replace_colnum(p,j-mf,q[:,i]))/detp for j in range(mf,m)])
            elif k<mf and i>=mf:
                th[k,i] = sum([Om1[k,j]*det(replace_colnum(p,j-mf,l[:,i-mf]))/detp for j in range(mf,m)])
            elif k>=mf and i<mf:
                th[k,i] = det(replace_colnum(p,k-mf,q[:,i]))/detp
            else:
                th[k,i] = det(replace_colnum(p,k-mf,l[:,i-mf]))/detp
    return th


def merge(Om1,Om2,mf):
    m = Om1.shape[0]
    th = theta(Om1,Om2,mf)
    Om3 = np.zeros((m,m))
    for k in range(m):
        for i in range(m):
            if k<mf and i<mf:
                Om3[k,i] = sum([Om2[k,j]*th[j,i] for j in range(mf)])
            elif k<mf and i>=mf:
                Om3[k,i] = sum([Om2[k,j]*th[j,i] for j in range(mf)])+Om2[k,i]
            elif k>=mf and i<mf:
                Om3[k,i] = sum([Om1[k,j]*th[j,i] for j in range(mf,m)])+Om1[k,i]
            else:
                Om3[k,i] = sum([Om1[k,j]*th[j,i] for j in range(mf,m)])
    return Om3

def scatter_matrix(m,id1,id2,value):
    id1,id2 = int(id1),int(id2)
    matrix = np.eye(m)
    matrix[id1,id1] = 1-value/2
    matrix[id1,id2] = value/2
    matrix[id2,id2] = 1-value/2
    matrix[id2,id1] = value/2
    return matrix


def master_matrix(num_term,mf,effective_matrices):
    mat = np.zeros(num_term)
    m = effective_matrices[0].shape[0]
    for t in range(1,num_term+1):
        premat,aftmat = effective_matrices[t-1], effective_matrices[t]
        mat[t,t] = num_term-premat[:mf,mf:].sum()-aftmat[mf:,:mf].sum()
        mat[t,t-1] = premat[:mf,:mf].sum()
        mat[t,t+1] = aftmat[mf:,mf:].sum()

#====================================================================================


#------------------------------------------------------------------------------------
# helper functions
# TODO: build this function to generate a sequence with given types and values

def generate_bynumber(message):
    '''
    A "message" should be formatted into a nested list structure, e.g.,
    [[0,1,0.4,3],[0,2,0.3,5],[1,2,0.5,3]]. Inside the first item [0,1,0.4,3], the first
    two items 0 and 1 tell which two edge states participate, the third item 0.4 gives
    the value for this matrix, the last item 3 represents the number of scattering events.
    '''
    seq_in_list = []
    for m in message:
        seq_in_list.extend([m[:3]]*m[3])
        random.shuffle(seq_in_list)
    return np.array(seq_in_list)

def fusion(seq,m,mf):
    # Create an empty list to store matrices that will be constructed based on the input sequence 'seq'.
    matrices = []
    # Iterate through each element (interaction type 't' and strength 'v') in 'seq'.
    for id1,id2,v in zip(seq[:, 0],seq[:, 1],seq[:,2]):
        matrix = scatter_matrix(m,id1,id2,v)
        matrices.append(matrix)
    # Initialize the result matrix 'mat0' with the first matrix in the 'matrices' list.
    mat0 = matrices[0]

    # Merge the matrices in 'matrices' list sequentially and update 'mat0' with the result.
    for mat1 in matrices[1:]:
        res = merge(mat0, mat1, mf)
        mat0 = res
    # Return the final merged matrix 'res' and the input sequence 'seq'.
    return res, seq

def states_check(seq,init_state,mf):
    # Create an empty list to store matrices that will be constructed based on the input sequence 'seq'.
    matrices = []
    ths = []
    m = len(init_state)
    states = np.zeros([m, len(seq)+1])
    # Iterate through each element (interaction type 't' and strength 'v') in 'seq'.
    for id1,id2,v in zip(seq[:, 0],seq[:, 1],seq[:,2]):
        matrix = scatter_matrix(m,id1,id2,v)
        matrices.append(matrix)

    # Initialize the result matrix 'mat0' with the first matrix in the 'matrices' list.
    mat0 = matrices[0]
    for mat1 in matrices[1:]:
        omega = merge(mat0, mat1, mf)
        ths.append(theta(mat0, mat1, mf))
        mat0 = omega # omega connects the initial and final states by the end of this for-loop.

    end_state = np.dot(omega, init_state)
    temp_state = copy.deepcopy(init_state)
    # calculate all the states between initial and final states
    for i, th in enumerate(ths[::-1]):
        newstate= np.dot(th,temp_state)
        states[:,-i-2] = newstate
        temp_state[2] = newstate[2]

    # connect intermediate states with initial and final states
    for i in range(mf):
        states[i, 0] = init_state[i]
        states[i,-1] = end_state[i]
    for j in range(mf,m):
        states[j,-1] = init_state[j]
        states[j,0] = end_state[j]
    return states, seq

