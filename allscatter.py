import numpy as np
import copy
import random
import matplotlib.pyplot as plt


# TODO: add this class to create a system consisting of electron reserviors (contacts) and interactions in between
class system:
    def __init__(self,current,diagram,mf,ground_term,blocking_state = None):
        ''' This diagram should be a list containing effective_matrix instances between adjacent contacts
        along the forward-propagation direction.
        '''
        self.term_current = current
        self.diagram = diagram
        self.m = diagram[0].trans_mat().shape[0]
        self.mf = mf
        self.num_term = len(diagram)
        self.ground_term = ground_term
        self.blocking_state = blocking_state
    def mastermat(self):
        blocking_state = self.blocking_state
        effective_matrices = self.diagram
        num_term = self.num_term
        mf = self.mf
        mat = np.zeros((num_term,num_term))
        for t in range(num_term):
            premat, aftmat = effective_matrices[self.prev(t)].trans_mat(), effective_matrices[t].trans_mat()
            mat[t, t] = self.m - premat[:mf, mf:].sum() - aftmat[mf:, :mf].sum()
            mat[t, self.prev(t)] = -premat[:mf, :mf].sum()
            mat[t, self.after(t)] = -aftmat[mf:, mf:].sum()
        if blocking_state is not None:
            id_terms, id_edges= [info[0] for info in blocking_state], [info[1] for info in blocking_state]
            for t in id_terms:
                # make corrections to the matrix elements connecting central terminal t and adjacent terminals t-1 and t+1.
                idt = id_terms.index(t)
                premat, aftmat = effective_matrices[self.prev(t)].trans_mat(), effective_matrices[t].trans_mat()
                mat[t, t] = mat[t, t]+sum([premat[id,mf:].sum() if id<mf else aftmat[id,:mf].sum() for id in id_edges[idt]])-len(id_edges[idt])
                mat[t, self.prev(t)] = mat[t, self.prev(t)]+sum([premat[id,:mf].sum() if id<mf else 0 for id in id_edges[idt]])
                mat[t, self.after(t)] = mat[t, self.after(t)]+sum([aftmat[id,mf:].sum() if id>=mf else 0 for id in id_edges[idt]])
            # make corrections to the matrix elements connecting non-adjacent terminals due to the blocked edge states.
            edges,table = self.which_terminal()
            for edge in edges:
                entry_term = [i for i in range(self.num_term)]
                for relation in table:
                    if int(relation[0]) == edge:
                        entry_term.remove(int(relation[1]))
                entry_term = sorted(entry_term)
                if edge < mf:
                    for i, term in enumerate(entry_term):
                        premat = effective_matrices[self.prev(term)].trans_mat()
                        i_prev = self.prev(i,len(entry_term))
                        if entry_term[i_prev] is not self.prev(term):
                            merge_mat = effective_matrices[entry_term[i_prev]].trans_mat()
                            current_term = entry_term[i_prev]
                            while self.after(current_term) is not term:
                                merge_mat = merge(merge_mat,effective_matrices[self.after(current_term)].trans_mat(),mf)
                                current_term = self.after(current_term)
                            mat[term,term] = mat[term,term]+premat[edge, mf:].sum()-merge_mat[edge,mf:].sum()
                            mat[term, self.prev(term)] = mat[term, self.prev(term)] + premat[edge, :mf].sum()
                            mat[term,entry_term[i_prev]] = mat[term, entry_term[i_prev]]-merge_mat[edge,:mf].sum()
                else:
                    for i, term in enumerate(entry_term):
                        aftmat = effective_matrices[term].trans_mat()
                        i_after = self.after(i,len(entry_term))
                        if entry_term[i_after] is not self.after(term):
                            merge_mat = effective_matrices[term].trans_mat()
                            current_term = term
                            while self.after(current_term) is not entry_term[i_after]:
                                merge_mat = merge(merge_mat,effective_matrices[self.after(current_term)].trans_mat(),mf)
                                current_term = self.after(current_term)
                            mat[term, term] = mat[term, term]+aftmat[edge,:mf].sum()-merge_mat[edge,:mf].sum()
                            mat[term, self.after(term)] = mat[term, self.after(term)] + aftmat[edge, mf:].sum()
                            mat[term,entry_term[i_after]]=mat[term,entry_term[i_after]]-merge_mat[edge,mf:].sum()
        return mat
    def which_terminal(self):
        blocking_state = self.blocking_state
        if blocking_state is None:
            return None
        id_term, id_edges = [info[0] for info in blocking_state], [info[1] for info in blocking_state]
        edge_term_table=[]
        scattered_edges = []
        for i, edges in enumerate(id_edges):
            for edge in edges:
                edge_term_table.append([edge,id_term[i]])
                if edge not in scattered_edges:
                    scattered_edges.append(edge)
        return scattered_edges, edge_term_table
    def prev(self,id,period=None):
        if period is None:
            period = self.num_term
        if int(id)==0:
            return int(period-1)
        else:
            return int(id-1)
    def after(self,id,period=None):
        if period is None:
            period = self.num_term
        if int(id)==(period-1):
            return 0
        else:
            return int(id+1)
    def solve(self):
        mf = self.mf
        ground_term = self.ground_term
        term_voltages = np.linalg.solve(self.mastermat(),self.term_current)
        return term_voltages-term_voltages[ground_term]
    def voltage_tracker(self):
        mf = self.mf
        term_voltages = self.solve()
        states = []
        m = self.m
        for i, eff_mat in enumerate(self.diagram):
            init_state = []
            init_state.extend([term_voltages[i]]*mf)
            if i == self.num_term-1:
                init_state.extend([term_voltages[0]]*(m-mf))
            else:
                init_state.extend([term_voltages[i+1]]*(m-mf))
            states.append(eff_mat.status_check(init_state))
        return states,term_voltages
    def voltage_plot(self,probe_width):
        m = self.m
        mf = self.mf
        states,term_voltages  = self.voltage_tracker()
        pre_state = np.hstack((np.ones((m, probe_width)) * term_voltages[0], states[0]))
        state_throughout = []
        for state, term in zip(states[1:], term_voltages[1:]):
            state_throughout = np.hstack((np.hstack((pre_state, np.ones((m, probe_width)) * term)), state))
            pre_state = state_throughout
        [plt.plot(edgestate,'r') for edgestate in state_throughout[:mf]]
        [plt.plot(edgestate,'b') for edgestate in state_throughout[mf:]]

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
    def status_check(self,init_state):
        mf = self.mf
        seq = self.seq
        # Create an empty list to store matrices that will be constructed based on the input sequence 'seq'.
        matrices = []
        ths = []
        m = len(init_state)
        states = np.zeros([m, len(seq) + 1])
        # Iterate through each element (interaction type 't' and strength 'v') in 'seq'.
        for id1, id2, v in zip(seq[:, 0], seq[:, 1], seq[:, 2]):
            matrix = scatter_matrix(m, id1, id2, v)
            matrices.append(matrix)

        # Initialize the result matrix 'mat0' with the first matrix in the 'matrices' list.
        mat0 = matrices[0]
        for mat1 in matrices[1:]:
            omega = merge(mat0, mat1, mf)
            ths.append(theta(mat0, mat1, mf))
            mat0 = omega  # omega connects the initial and final states by the end of this for-loop.

        end_state = np.dot(omega, init_state)
        temp_state = copy.deepcopy(init_state)
        # calculate all the states between initial and final states
        for i, th in enumerate(ths[::-1]):
            newstate = np.dot(th, temp_state)
            states[:, -i - 2] = newstate
            temp_state[2] = newstate[2]

        # connect intermediate states with initial and final states
        for i in range(mf):
            states[i, 0] = init_state[i]
            states[i, -1] = end_state[i]
        for j in range(mf, m):
            states[j, -1] = init_state[j]
            states[j, 0] = end_state[j]
        return states
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

