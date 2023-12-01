import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import warnings

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
        edges = [edge.trans_mat() for edge in self.graph]
        numTerminal = self.numTerminal
        nfm = self.numForwardMover
        tnm = self.totalNumMover
        mat = np.zeros((numTerminal,numTerminal))
        for t in range(numTerminal):
            premat, aftmat = edges[self.prev(t)], edges[t]
            mat[t, t] = self.totalNumMover - premat[:nfm, nfm:].sum() - aftmat[nfm:, :nfm].sum()
            mat[t, self.prev(t)] = -premat[:nfm, :nfm].sum()
            mat[t, self.after(t)] = -aftmat[nfm:, nfm:].sum()
        # print("before correction\n",mat)
        if blockStates is not None:
            idscatteredEdge, edgeBlockedInThese = self.which_blocked()
            # print('idscatteredEdge', idscatteredEdge)
            # print('edgeBlockedInThese',edgeBlockedInThese)
            idTerms, idEdges= [info[0] for info in blockStates], [info[1] for info in blockStates]
            terminals = np.arange(0, numTerminal, 1, dtype=int).tolist()
            fullset = np.arange(0, tnm, 1, dtype=int).tolist()
            table = [[term,list(set(fullset)-set(idEdges[idTerms.index(term)]))] if term in idTerms else [term,fullset] for term in terminals]
            # print("idTerms",idTerms)
            # print("idEdges", idTerms)
            # print("table",table)
            for t in idTerms:
                # make corrections to the matrix elements connecting central terminal t and adjacent terminals prev(t) and after(t).
                idt = idTerms.index(t)
                premat, aftmat = edges[self.prev(t)], edges[t]
                mat[t, t]+=sum([premat[index, nfm:].sum() if index<nfm else aftmat[index, :nfm].sum() for index in idEdges[idt]])-len(idEdges[idt])
                mat[t, self.prev(t)]+=sum([premat[index,:nfm].sum() if index<nfm else 0 for index in idEdges[idt]])
                mat[t, self.after(t)]+=sum([aftmat[index,nfm:].sum() if index>=nfm else 0 for index in idEdges[idt]])
            # make corrections to the matrix elements connecting non-adjacent terminals due to the blocked edge states.
            # print("after 1st correction\n",mat)
            for j in idscatteredEdge:
                # print("blocked j",j)
                jEnterThese = copy.deepcopy(terminals)
                for relation in edgeBlockedInThese:
                    if int(relation[0]) is j:
                        jEnterThese.remove(int(relation[1]))
                # print("jEnterThese", jEnterThese)
                if j<nfm:
                    # print("j<nfm",j)
                    for i, t in enumerate(jEnterThese):
                        i_prev = self.prev(i,len(jEnterThese))
                        if jEnterThese[i_prev] is not self.prev(t):
                            # print("t=",t)
                            mat[t,t] += edges[self.prev(t)][j,nfm:].sum()
                            mat[t,self.prev(t)] += edges[self.prev(t)][j,:nfm].sum()
                            # print("after 1.5st correction\n",mat)
                            changes = self.muj_finalstate(j,t,table)
                            # print("changes",changes)
                            for term in terminals:
                                mat[t,term] -= changes[term]
                else:
                    # print("j>=nfm",j)
                    for i, t in enumerate(jEnterThese):
                        i_after = self.after(i,len(jEnterThese))
                        if jEnterThese[i_after] is not self.after(t):
                            mat[t,t] += edges[t][j,:nfm].sum()
                            mat[t,self.after(t)] += edges[t][j,nfm:].sum()
                            changes = self.muj_finalstate(j,self.after(t),table)
                            for term in terminals:
                                mat[t,term] -= changes[term]
        return mat
    def which_blocked(self):
        blockStates = self.blockStates
        if blockStates is None:
            return None
        idTerm, idEdges = [info[0] for info in blockStates], [info[1] for info in blockStates]
        edgeBlockedInThese=[]
        blockedEdges = []
        for i, edges in enumerate(idEdges):
            for edge in edges:
                edgeBlockedInThese.append([edge,idTerm[i]])
                if edge not in blockedEdges:
                    blockedEdges.append(edge)
        return blockedEdges, edgeBlockedInThese
    def muj_finalstate(self, j, t, table):
        matrices = [edge.trans_mat() for edge in self.graph]
        nfm = self.numForwardMover
        tnm = self.totalNumMover
        ntm = self.numTerminal
        changes = [0] * ntm
        fullset = np.arange(0, tnm, 1, dtype=int).tolist()
        terminals = np.arange(0, ntm, 1, dtype=int).tolist()
        for k in [s for s in table[self.prev(t)][1] if s < nfm]:
            changes[self.prev(t)] += matrices[self.prev(t)][j, k]  # First term
        for k in [s for s in table[t][1] if s >= nfm]:
            changes[t] += matrices[self.prev(t)][j, k]  # Second term
        for k in [s for s in list(set(fullset) - set(table[self.prev(t)][1])) if s < nfm]:
            chgSubpre = self.muj_finalstate(k, self.prev(t), table)  # Third term
            for term in terminals:
                changes[term] += matrices[self.prev(t)][j, k] * chgSubpre[term]
        for k in [s for s in list(set(fullset) - set(table[t][1])) if s >= nfm]:
            chgSubaft = self.muj_finalstate(k, self.after(t), table) # Fourth term
            for term in terminals:
                changes[term] += matrices[self.prev(t)][j, k] * chgSubaft[term]
        return changes
    def prev(self,index,period=None):
        if period is None:
            period = self.numTerminal
        if int(index)==0:
            return int(period-1)
        else:
            return int(index-1)
    def after(self,index,period=None):
        if period is None:
            period = self.numTerminal
        if int(index)==(period-1):
            return 0
        else:
            return int(index+1)
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
    def plot(self,figsize=(12,10)):
        tnm = self.totalNumMover
        nfm = self.numForwardMover
        numTerminal = self.numTerminal
        blockStates = self.blockStates
        edges = self.graph
        _, axs = plt.subplots(2, len(edges), figsize=figsize,sharex=True,sharey=True)
        termVoltages = self.solve()
        plt.subplots_adjust(hspace=0.1)
        if blockStates is not None:
            idTerms, idEdges = [info[0] for info in blockStates], [info[1] for info in blockStates]
            terminals = np.arange(0, numTerminal, 1, dtype=int).tolist()
            fullset = np.arange(0, tnm, 1, dtype=int).tolist()
            table = [
                [term, list(set(fullset) - set(idEdges[idTerms.index(term)]))] if term in idTerms else [term, fullset]
                for term in terminals]
        try:
            for t, edge in enumerate(edges):
                initStates = [termVoltages[t] if j<nfm else termVoltages[self.after(t)] for j in range(tnm)]
                if blockStates is not None:
                    if t in idTerms:
                        for j in idEdges[idTerms.index(t)]:
                            if j<nfm:
                                initStates[j] = np.dot(self.muj_finalstate(j, t, table),termVoltages)
                            else:
                                initStates[j] = np.dot(self.muj_finalstate(j, self.after(t), table),termVoltages)
                edge.plot(initStates, ax1=axs[0,t], ax2=axs[1,t])
                if max(termVoltages)+0.1>1:
                    axs[1, 0].set_ylim(-0.1, max(termVoltages) + 0.1)
                else:
                    axs[1, 0].set_ylim(-0.1, 1.05)
                axs[1, t].axhline(y=termVoltages[t],xmin=0,xmax=0.4,linestyle='-',color='y')
                axs[1, t].axhline(y=termVoltages[self.after(t)], xmin=0.6, xmax=1, linestyle='-', color='y')
            return axs
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
    def plot(self,initStates,ax1=None,ax2=None):
        import matplotlib.patches as patches
        tnm = self.totalNumMover
        nfm = self.numForwardMover
        if ax1 is None or ax2 is None:
            _, axs = plt.subplots(2,1,figsize=(8,6),sharex=True)
            ax1, ax2 = axs
            plt.subplots_adjust(hspace=0.1)
        seq = self.seq
        try:
            scatterSite = np.arange(0.5,len(seq)+0.5,1)
            scatterValue = [scatter[2] for scatter in seq]
            colorsUpper = [[(nfm-scatter[0])/tnm,0.1,0] if scatter[0]<nfm else [0,0.1,(scatter[0]+1)/tnm] for scatter in seq]
            colorsLower = [[(nfm-scatter[1])/tnm,0.1,0] if scatter[1]<nfm else [0,0.1,(scatter[1]+1)/tnm] for scatter in seq]
            for xi, yi, colorUpper,colorLower in zip(scatterSite,scatterValue,colorsUpper,colorsLower):
                # Define a circle radius and center
                radius = 0.05
                circle_center = (xi, yi)
                # Create upper half (red)
                theta1, theta2 = 0, 180  # Degrees
                upper_half = patches.Wedge(circle_center, radius, theta1, theta2, color=colorUpper)
                # Create lower half (blue)
                theta1, theta2 = 180, 360  # Degrees
                lower_half = patches.Wedge(circle_center, radius, theta1, theta2, color=colorLower)
                # Add patches to the axes
                ax1.add_patch(upper_half)
                ax1.add_patch(lower_half)
            states = self.status_check(initStates)
            [ax2.plot(states[row,:],color=[(nfm-row)/tnm,0.1,0]) if row<nfm else ax2.plot(states[row,:],color=[0,0.1,(row+1)/tnm]) for row in range(tnm)]
            ax1.set_ylabel('Value')
            ax1.set_ylim(-0.05,1.05)
            ax2.set_xlabel('Interaction Site')
            ax2.set_ylabel('Flow Status')
            return (ax1, ax2)
        except:
            return False
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

