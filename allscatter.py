import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import warnings

# TODO: add this class to create a System consisting of electron reserviors (contacts) and interactions in between
class System:
    def __init__(self,nodesCurrent,graph,numForwardMover,zeroVoltTerminal,blockStates = None):
        ''' This graph should be a list containing Edge instances between adjacent contacts
        along the forward-propagation direction.
        '''
        self.nodesCurrent = nodesCurrent
        self.graph = graph
        self.totalNumMover = graph[0].trans_mat().shape[0]
        self.numForwardMover= numForwardMover
        self.numTerminal = len(graph)
        self.zeroVoltTerminal = zeroVoltTerminal
        self.blockStates = blockStates
    def mastermat(self):
        blockStates = self.blockStates
        edges = self.graph
        numTerminal = self.numTerminal
        nfm = self.numForwardMover
        mat = np.zeros((numTerminal,numTerminal))
        for t in range(numTerminal):
            premat, aftmat = edges[self.prev(t)].trans_mat(), edges[t].trans_mat()
            mat[t, t] = self.totalNumMover - premat[:nfm, nfm:].sum() - aftmat[nfm:, :nfm].sum()
            mat[t, self.prev(t)] = -premat[:nfm, :nfm].sum()
            mat[t, self.after(t)] = -aftmat[nfm:, nfm:].sum()
        if blockStates is not None:
            idTerms, idEdges= [info[0] for info in blockStates], [info[1] for info in blockStates]
            for t in idTerms:
                # make corrections to the matrix elements connecting central terminal t and adjacent terminals t-1 and t+1.
                idt = idTerms.index(t)
                premat, aftmat = edges[self.prev(t)].trans_mat(), edges[t].trans_mat()
                mat[t, t] = mat[t, t]+sum([premat[id, nfm:].sum() if id<nfm else aftmat[id, :nfm].sum() for id in idEdges[idt]])-len(idEdges[idt])
                mat[t, self.prev(t)] = mat[t, self.prev(t)]+sum([premat[id,:nfm].sum() if id<nfm else 0 for id in idEdges[idt]])
                mat[t, self.after(t)] = mat[t, self.after(t)]+sum([aftmat[id,nfm:].sum() if id>=nfm else 0 for id in idEdges[idt]])
            # make corrections to the matrix elements connecting non-adjacent terminals due to the blocked edge states.
            edges,table = self.which_terminal()
            for edge in edges:
                entry_term = [i for i in range(self.numTerminal)]
                for relation in table:
                    if int(relation[0]) == edge:
                        entry_term.remove(int(relation[1]))
                entry_term = sorted(entry_term)
                if edge < nfm:
                    for i, term in enumerate(entry_term):
                        premat = edges[self.prev(term)].trans_mat()
                        i_prev = self.prev(i,len(entry_term))
                        if entry_term[i_prev] is not self.prev(term):
                            merge_mat = edges[entry_term[i_prev]].trans_mat()
                            current_term = entry_term[i_prev]
                            while self.after(current_term) is not term:
                                merge_mat = merge(merge_mat,edges[self.after(current_term)].trans_mat(), nfm)
                                current_term = self.after(current_term)
                            mat[term,term] = mat[term,term]+premat[edge, nfm:].sum()-merge_mat[edge, nfm:].sum()
                            mat[term, self.prev(term)] = mat[term, self.prev(term)] + premat[edge, :nfm].sum()
                            mat[term,entry_term[i_prev]] = mat[term, entry_term[i_prev]]-merge_mat[edge,:nfm].sum()
                else:
                    for i, term in enumerate(entry_term):
                        aftmat = edges[term].trans_mat()
                        i_after = self.after(i,len(entry_term))
                        if entry_term[i_after] is not self.after(term):
                            merge_mat = edges[term].trans_mat()
                            current_term = term
                            while self.after(current_term) is not entry_term[i_after]:
                                merge_mat = merge(merge_mat,edges[self.after(current_term)].trans_mat(),nfm)
                                current_term = self.after(current_term)
                            mat[term, term] = mat[term, term]+aftmat[edge,:nfm].sum()-merge_mat[edge,:nfm].sum()
                            mat[term, self.after(term)] = mat[term, self.after(term)] + aftmat[edge, nfm:].sum()
                            mat[term,entry_term[i_after]]=mat[term,entry_term[i_after]]-merge_mat[edge,nfm:].sum()
        return mat
    def which_terminal(self):
        blockStates = self.blockStates
        if blockStates is None:
            return None
        idTerm, idEdges = [info[0] for info in blockStates], [info[1] for info in blockStates]
        edge_term_table=[]
        scattered_edges = []
        for i, edges in enumerate(idEdges):
            for edge in edges:
                edge_term_table.append([edge,idTerm[i]])
                if edge not in scattered_edges:
                    scattered_edges.append(edge)
        return scattered_edges, edge_term_table
    def prev(self,id,period=None):
        if period is None:
            period = self.numTerminal
        if int(id)==0:
            return int(period-1)
        else:
            return int(id-1)
    def after(self,id,period=None):
        if period is None:
            period = self.numTerminal
        if int(id)==(period-1):
            return 0
        else:
            return int(id+1)
    def solve(self):
        nfm = self.numForwardMover
        termVoltages = np.array([0]*self.numTerminal)
        zeroVoltTerminal = self.zeroVoltTerminal
        if np.linalg.det(self.mastermat()) != 0:
            termVoltages = np.linalg.solve(self.mastermat(),self.nodesCurrent)
        else:
            warnings.warn("Your matrix is singular. `np.linalg.lstsq` is used to find an approximate solution.")
            termVoltages,_,_,_ = np.linalg.lstsq(self.mastermat(),self.nodesCurrent,rcond=None)
        return termVoltages-termVoltages[zeroVoltTerminal]
    def voltage_tracker(self):
        nfm = self.numForwardMover
        termVoltages = self.solve()
        states = []
        tnm= self.totalNumMover
        for i, effMat in enumerate(self.graph):
            initState = []
            initState.extend([termVoltages[i]]*nfm)
            if i == self.numTerminal-1:
                initState.extend([termVoltages[0]]*(tnm-nfm))
            else:
                initState.extend([termVoltages[i+1]]*(tnm-nfm))
            states.append(effMat.status_check(initState))
        return states,termVoltages
    def voltage_plot(self,probeWidth=5):
        tnm = self.totalNumMover
        nfm = self.numForwardMover
        states, termVoltages  = self.voltage_tracker()
        preStatus = np.hstack((np.ones((tnm, probeWidth)) * termVoltages[0], states[0]))
        allStatus = []
        for state, term in zip(states[1:], termVoltages[1:]):
            allStatus = np.hstack((np.hstack((preStatus, np.ones((tnm, probeWidth)) * term)), state))
            preStatus = allStatus
        try:
            [plt.plot(edgeStatus,'r') for edgeStatus in allStatus[:nfm]]
            if nfm<tnm: # There exist one or more backward movers
                [plt.plot(edgeStatus,'b') for edgeStatus in allStatus[nfm:]]
            return True
        except:
            return False

class Edge:
    def __init__(self,sequence,totalNumMover,numForwardMover):
        self.seq = sequence
        self.totalNumMover = totalNumMover
        self.numForwardMover = numForwardMover
    def get_seq(self):
        return self.seq
    def trans_mat(self):
        seq = self.seq
        tnm = self.totalNumMover
        nfm = self.numForwardMover
        matrices = []
        # Iterate through each element (interaction type 't' and strength 'v') in 'seq'.
        for id1, id2, v in zip(seq[:, 0], seq[:, 1], seq[:, 2]):
            matrix = scatter_matrix(tnm, id1, id2, v)
            matrices.append(matrix)
        # Initialize the result matrix 'mat0' with the first matrix in the 'matrices' list.
        mat0 = matrices[0]
        # Forward-propagation process: calculate all transformation parameters
        for mat1 in matrices[1:]:
            res = merge(mat0, mat1, nfm)
            mat0 = res
        # Return the final merged matrix 'res' and the input sequence 'seq'.
        return res
    def status_check(self,initState):
        nfm = self.numForwardMover
        seq = self.seq
        # Create an empty list to store matrices that will be constructed based on the input sequence 'seq'.
        matrices = []
        thetaMatrices = []
        tnm = self.totalNumMover
        states = np.zeros([tnm, len(seq) + 1])
        # Iterate through each element (interaction type 't' and strength 'v') in 'seq'.
        for id1, id2, v in zip(seq[:, 0], seq[:, 1], seq[:, 2]):
            matrix = scatter_matrix(tnm, id1, id2, v)
            matrices.append(matrix)

        # Initialize the result matrix 'mat0' with the first matrix in the 'matrices' list.
        mat0 = matrices[0]
        # Forward-propagation process: calculate all transformation parameters
        for mat1 in matrices[1:]:
            omega = merge(mat0, mat1, nfm)
            thetaMatrices.append(theta(mat0, mat1, nfm))
            mat0 = omega  # omega connects the initial and final states by the end of this for-loop.
        finalState = np.dot(omega, initState)
        tempState = copy.deepcopy(initState)
        # Back-propagation process: calculate all intermediate states
        for i, thetaMatrix in enumerate(thetaMatrices[::-1]):
            newState = np.dot(thetaMatrix, tempState)
            states[:, -i - 2] = newState
            if nfm<tnm:
                tempState[nfm:] = newState[nfm:]
            else:
                tempState = newState

        # connect intermediate states with initial and final states
        for i in range(nfm):
            states[i, 0] = initState[i]
            states[i, -1] = finalState[i]
        for j in range(nfm, tnm):
            states[j, -1] = initState[j]
            states[j, 0] = finalState[j]
        return states
#================================================================
#core functions

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

def scatter_matrix(m,id1,id2,value):
    id1,id2 = int(id1),int(id2)
    matrix = np.eye(m)
    matrix[id1,id1] = 1-value/2
    matrix[id1,id2] = value/2
    matrix[id2,id2] = 1-value/2
    matrix[id2,id1] = value/2
    return matrix


def master_matrix(numTerminal,nfm,edges):
    mat = np.zeros(numTerminal)
    m = edges[0].shape[0]
    for t in range(1,numTerminal+1):
        premat,aftmat = edges[t-1], edges[t]
        mat[t,t] = numTerminal-premat[:nfm,nfm:].sum()-aftmat[nfm:,:nfm].sum()
        mat[t,t-1] = premat[:nfm,:nfm].sum()
        mat[t,t+1] = aftmat[nfm:,nfm:].sum()

#====================================================================================


#------------------------------------------------------------------------------------
# helper functions
# TODO: build this function to generate a sequence with given types and values

def generate_bynumber(messages):
    '''
    A "message" should be formatted into a nested list structure, e.g.,
    [[0,1,0.4,3],[0,2,0.3,5],[1,2,0.5,3]]. Inside the first item [0,1,0.4,3], the first
    two items 0 and 1 tell which two edge states participate, the third item 0.4 gives
    the value for this matrix, the last item 3 represents the number of scattering events.
    '''
    seq_in_list = []
    for message in messages:
        seq_in_list.extend([message[:3]]*message[3])
        random.shuffle(seq_in_list)
    return np.array(seq_in_list)

def fusion(seq,m,nfm):
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
        res = merge(mat0, mat1, nfm)
        mat0 = res
    # Return the final merged matrix 'res' and the input sequence 'seq'.
    return res, seq

def states_check(seq,initState,nfm):
    # Create an empty list to store matrices that will be constructed based on the input sequence 'seq'.
    matrices = []
    ths = []
    m = len(initState)
    states = np.zeros([m, len(seq)+1])
    # Iterate through each element (interaction type 't' and strength 'v') in 'seq'.
    for id1,id2,v in zip(seq[:, 0],seq[:, 1],seq[:,2]):
        matrix = scatter_matrix(m,id1,id2,v)
        matrices.append(matrix)

    # Initialize the result matrix 'mat0' with the first matrix in the 'matrices' list.
    mat0 = matrices[0]
    omega = mat0
    for mat1 in matrices[1:]:
        omega = merge(mat0, mat1, nfm)
        ths.append(theta(mat0, mat1, nfm))
        mat0 = omega # omega connects the initial and final states by the end of this for-loop.

    finalState = np.dot(omega, initState)
    tempState = copy.deepcopy(initState)
    # calculate all the states between initial and final states
    for i, th in enumerate(ths[::-1]):
        newState= np.dot(th,tempState)
        states[:,-i-2] = newState
        tempState[2] = newState[2]

    # connect intermediate states with initial and final states
    for i in range(nfm):
        states[i, 0] = initState[i]
        states[i,-1] = finalState[i]
    for j in range(nfm,m):
        states[j,-1] = initState[j]
        states[j,0] = finalState[j]
    return states, seq

