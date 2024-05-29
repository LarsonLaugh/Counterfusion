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

import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt
from .algorithm import (
    theta,
    merge,
    interaction_builder,
)
from .utils import (
    check_blockstates_structure,
    check_blockstates_value,
    check_sequence_structure,
    check_sequence_value,
    system_to_json,
    edge_to_json,
)

__all__ = ["System", "Edge", "system_input_filter", "edge_input_filter"]


# ====================================================================================================
# Define the System class


class System:
    def __init__(
        self, nodesCurrent, graph, numForwardMover, zeroVoltTerminal, blockStates=None
    ):
        """
        Initialize a new instance of the System class.

        Args:
            nodesCurrent (np.array): The current values at each node in the system.
            graph (list): List of edge objects that describe the system's connectivity
              and behavior.
            numForwardMover (int): The number of forward movers in the system's states.
            zeroVoltTerminal (int): The index of the terminal with zero voltage, used
            as a reference.
            blockStates (list, optional): States that are blocked from changing,
            possibly due to external
            constraints.

        The constructor also performs input validation using `system_input_filter` to
        ensure valid initial settings.
        """
        # Validate the initial setup for the system's configuration
        if system_input_filter(
            nodesCurrent,
            graph,
            numForwardMover,
            graph[0].trans_mat().shape[0],
            zeroVoltTerminal,
            blockStates,
        ):
            self.nodesCurrent = nodesCurrent  # Current at nodes
            self.graph = graph  # Graph of edges
            self.totalNumMover = graph[0].trans_mat().shape[0]  # Total number of movers
            self.numForwardMover = numForwardMover  # Number of forward movers
            self.numTerminal = len(graph)  # Number of terminals or nodes in the system
            self.zeroVoltTerminal = (
                zeroVoltTerminal  # Reference terminal with zero voltage
            )
            self.blockStates = blockStates  # States that are blocked
        else:
            raise ValueError("Does not pass system_input_filter()")

    def mastermat(self):
        """
        Calculates the master matrix based on the system's configuration which is
        used to solve the system equations.
        """
        blockStates = self.blockStates
        edges = [edge.trans_mat() for edge in self.graph]
        numTerminal = self.numTerminal
        nfm = self.numForwardMover
        tnm = self.totalNumMover
        mat = np.zeros((numTerminal, numTerminal))

        # Handling blocked states and calculating the matrix entries accordingly
        if blockStates is None:
            # Populate matrix for simple case without blocked states
            for t in range(numTerminal):
                premat, aftmat = edges[self._prev(t)], edges[t]
                mat[t, t] = tnm - premat[:nfm, nfm:].sum() - aftmat[nfm:, :nfm].sum()
                if numTerminal > 2:
                    mat[t, self._prev(t)] = -premat[:nfm, :nfm].sum()
                    mat[t, self._after(t)] = -aftmat[nfm:, nfm:].sum()
                else:
                    mat[t, self._prev(t)] = (
                        -premat[:nfm, :nfm].sum() - aftmat[nfm:, nfm:].sum()
                    )
        else:
            # Handle blocked states by adjusting matrix based on the allowed changes
            idTerms, idEdges = [info[0] for info in blockStates], [
                info[1] for info in blockStates
            ]
            terminals = np.arange(0, numTerminal, 1, dtype=int).tolist()
            fullset = np.arange(0, tnm, 1, dtype=int).tolist()
            table = [
                (
                    [term, list(set(fullset) - set(idEdges[idTerms.index(term)]))]
                    if term in idTerms
                    else [term, fullset]
                )
                for term in terminals
            ]
            for t in range(numTerminal):
                for k in table[t][1]:
                    if k < nfm:
                        changes = self._muj_finalstate(k, t, table)
                    else:
                        changes = self._muj_finalstate(k, self._after(t), table)
                    for term in terminals:
                        mat[t, term] -= changes[term]
                mat[t, t] += len(table[t][1])
        return mat

    def solve(self):
        """
        Solve the system equations using the computed master matrix.
        """
        zeroVoltTerminal = self.zeroVoltTerminal
        if np.linalg.det(self.mastermat()) != 0:
            # Solve using a standard linear algebra approach if the matrix is not singular
            termVoltages = np.linalg.solve(self.mastermat(), self.nodesCurrent)
        else:
            # Use least squares solution if the matrix is singular
            warnings.warn(
                "Your matrix is singular. `np.linalg.lstsq` is used to find an approximate solution."
            )
            termVoltages, _, _, _ = np.linalg.lstsq(
                self.mastermat(), self.nodesCurrent, rcond=None
            )
        # Normalize voltages by setting the zero-volt terminal voltage to zero
        return termVoltages - termVoltages[zeroVoltTerminal]

    def plot(self, figsize=(12, 10)):
        """
        Visualizes the state evolution and interactions between terminals using matplotlib.

        Args:
            figsize (tuple): The dimensions of the figure to display the plots.

        Returns:
            matplotlib.axes.Axes or False: The plot axes if successful, False if an error occurs.
        """
        tnm = self.totalNumMover
        nfm = self.numForwardMover
        numTerminal = self.numTerminal
        blockStates = self.blockStates
        edges = self.graph
        _, axs = plt.subplots(2, len(edges), figsize=figsize, sharex=True, sharey=True)
        termVoltages = self.solve()
        plt.subplots_adjust(hspace=0.1)

        # Initialize and display state information for each terminal
        initStatesList = []
        for t in range(len(edges)):
            initStatesList.append(
                [
                    termVoltages[t] if j < nfm else termVoltages[self._after(t)]
                    for j in range(tnm)
                ]
            )

        if blockStates is not None:
            idTerms, idEdges = [info[0] for info in blockStates], [
                info[1] for info in blockStates
            ]
            terminals = np.arange(0, numTerminal, 1, dtype=int).tolist()
            fullset = np.arange(0, tnm, 1, dtype=int).tolist()
            table = [
                (
                    [term, list(set(fullset) - set(idEdges[idTerms.index(term)]))]
                    if term in idTerms
                    else [term, fullset]
                )
                for term in terminals
            ]
            for t in idTerms:
                for j in idEdges[idTerms.index(t)]:
                    if j < nfm:
                        initStatesList[t][j] = np.dot(
                            self._muj_finalstate(j, t, table), termVoltages
                        )
                    else:
                        initStatesList[self._prev(t)][j] = np.dot(
                            self._muj_finalstate(j, self._after(t), table), termVoltages
                        )
        try:
            for t, (initStates, edge) in enumerate(zip(initStatesList, edges)):
                edge.plot(initStates, ax1=axs[0, t], ax2=axs[1, t])
                if max(termVoltages) + 0.1 > 1:
                    axs[1, 0].set_ylim(-0.1, max(termVoltages) + 0.1)
                else:
                    axs[1, 0].set_ylim(-0.1, 1.05)
                axs[1, t].axhline(
                    y=termVoltages[t], xmin=0, xmax=0.4, linestyle="-", color="y"
                )
                axs[1, t].axhline(
                    y=termVoltages[self._after(t)],
                    xmin=0.6,
                    xmax=1,
                    linestyle="-",
                    color="y",
                )
            return axs
        except:
            print("Failed to plot!")
            return False

    def _muj_finalstate(self, j, t, table):
        """
        Recursively compute the final state contributions from different terminals.

        Args:
            j (int): Current state index.
            t (int): Terminal index.
            idEdges (list): Indices of edges corresponding to blocked states.
            idTerms (list): Indices of terminals with blocked states.

        Returns:
            list: Adjustments to the terminal voltages due to state j.
        """
        matrices = [edge.trans_mat() for edge in self.graph]
        nfm = self.numForwardMover
        tnm = self.totalNumMover
        ntm = self.numTerminal
        changes = [0] * ntm
        fullset = np.arange(0, tnm, 1, dtype=int).tolist()
        terminals = np.arange(0, ntm, 1, dtype=int).tolist()

        # Calculate contributions to previous terminal (using transitions into backward mover states)
        for k in [s for s in table[self._prev(t)][1] if s < nfm]:
            changes[self._prev(t)] += matrices[self._prev(t)][j, k]

        # Calculate contributions to current terminal (using transitions from forward mover states)
        for k in [s for s in table[t][1] if s >= nfm]:
            changes[t] += matrices[self._prev(t)][j, k]

        # Recursively calculate changes for states that are not in the table for the previous terminal
        for k in [
            s for s in list(set(fullset) - set(table[self._prev(t)][1])) if s < nfm
        ]:
            chgSubpre = self._muj_finalstate(k, self._prev(t), table)
            for term in terminals:
                changes[term] += matrices[self._prev(t)][j, k] * chgSubpre[term]

        # Recursively calculate changes for states that are not in the table for the current terminal
        for k in [s for s in list(set(fullset) - set(table[t][1])) if s >= nfm]:
            chgSubaft = self._muj_finalstate(k, self._after(t), table)
            for term in terminals:
                changes[term] += matrices[self._prev(t)][j, k] * chgSubaft[term]
        return changes

    def output_to_json(self, filename="data.json"):
        """
        Outputs the current state of the system into a JSON file, including node currents, graph information,
        and calculated voltages.
        """
        blockStates = self.blockStates
        numTerminal = self.numTerminal
        nodesCurrent = self.nodesCurrent
        nfm = self.numForwardMover
        tnm = self.totalNumMover
        edgeSequence = [edge.get_seq() for edge in self.graph]
        edgeMat = [edge.trans_mat() for edge in self.graph]
        initStatesList = []
        termVoltages = self.solve()

        # Initial state list generation for each terminal
        for t in range(numTerminal):
            initStatesList.append(
                [
                    termVoltages[t] if j < nfm else termVoltages[self._after(t)]
                    for j in range(tnm)
                ]
            )
        if blockStates is not None:
            idTerms, idEdges = [info[0] for info in blockStates], [
                info[1] for info in blockStates
            ]
            terminals = np.arange(0, numTerminal, 1, dtype=int).tolist()
            fullset = np.arange(0, tnm, 1, dtype=int).tolist()
            table = [
                (
                    [term, list(set(fullset) - set(idEdges[idTerms.index(term)]))]
                    if term in idTerms
                    else [term, fullset]
                )
                for term in terminals
            ]
            for t in idTerms:
                for j in idEdges[idTerms.index(t)]:
                    if j < nfm:
                        initStatesList[t][j] = np.dot(
                            self._muj_finalstate(j, t, table), termVoltages
                        )
                    else:
                        initStatesList[self._prev(t)][j] = np.dot(
                            self._muj_finalstate(j, self._after(t), table), termVoltages
                        )

        # Generating system state information
        stateInfo = [
            edge.status_check(initStates)
            for edge, initStates in zip(self.graph, initStatesList)
        ]
        sysMat = self.mastermat()
        blockStates = self.blockStates
        # Attempt to write all collected information to a JSON file
        try:
            system_to_json(
                filename,
                edgeSequence,
                edgeMat,
                stateInfo,
                nodesCurrent,
                sysMat,
                termVoltages,
                blockStates,
            )
        except:
            print("Error: Fail to write to " + filename)

    def _prev(self, index, period=None):
        """
        Calculate the previous terminal index in a circular system configuration.
        This method allows for cyclic interaction across the terminals, enabling a wrap-around effect.
        """
        if period is None:
            period = self.numTerminal
        if index == 0:
            return period - 1
        else:
            return index - 1

    def _after(self, index, period=None):
        """
        Calculate the next terminal index in a circular system configuration.
        Similar to `_prev`, this method supports cyclic operations but in the opposite direction.
        """
        if period is None:
            period = self.numTerminal
        if index == period - 1:
            return 0
        else:
            return index + 1


class Edge:
    def __init__(self, sequence, totalNumMover, numForwardMover):
        """
        Initialize an Edge object with sequence data and mover information.

        Args:
            sequence (np.array): Array of tuples (id1, id2, v) representing interactions between nodes.
            totalNumMover (int): Total number of movers (both forward and backward).
            numForwardMover (int): Number of forward movers.

        The constructor also includes input validation.
        """
        # Validate the input data before assigning it to the instance.
        if edge_input_filter(sequence, totalNumMover, numForwardMover):
            self.seq = sequence
            self.totalNumMover = totalNumMover
            self.numForwardMover = numForwardMover

    def get_seq(self):
        return self.seq

    def trans_mat(self):
        """
        Calculate the transition matrix for the edge based on the interaction sequence.

        Returns:
            np.array: Transition matrix which defines state transitions.
        """
        seq = self.seq
        tnm = self.totalNumMover
        nfm = self.numForwardMover
        matrices = []
        # Base case: if no interactions, return identity matrix of size totalNumMover
        if len(seq) == 0:
            return np.eye(tnm)
        # Build transition matrices for each interaction in the sequence
        for id1, id2, v in zip(seq[:, 0], seq[:, 1], seq[:, 2]):
            matrix = interaction_builder(tnm, id1, id2, v)
            matrices.append(matrix)
        # If only one matrix, no merging needed
        mat0 = matrices[0]
        # Forward-propagation process: calculate all transformation parameters
        if len(matrices) < 2:
            return mat0
        for mat1 in matrices[1:]:
            res = merge(mat0, mat1, nfm)
            mat0 = res
        return res

    def status_check(self, initStates):
        """
        Evaluate the state progression across the sequence interactions.

        Args:
            initStates (np.array): Initial state configuration.

        Returns:
            np.array: Array describing state transitions at each interaction step.
        """
        nfm = self.numForwardMover
        seq = self.seq
        matrices = []
        thetaMatrices = []
        tnm = self.totalNumMover
        states = np.zeros([tnm, len(seq) + 1])

        if len(seq) == 0:  # when there is no interaction
            for i in range(nfm):
                states[i, 0] = initStates[i]
                states[i, -1] = initStates[i]
            for j in range(nfm, tnm):
                states[j, -1] = initStates[j]
                states[j, 0] = initStates[j]
            return states

        for id1, id2, v in zip(seq[:, 0], seq[:, 1], seq[:, 2]):
            matrix = interaction_builder(tnm, id1, id2, v)
            matrices.append(matrix)
        mat0 = matrices[0]

        if nfm is tnm:  # When only forward-mover present.
            for i in range(nfm):
                states[i, 0] = initStates[i]
            for j, mat in enumerate(matrices):
                states[:, j + 1] = np.dot(mat, states[:, j])
            return states

        # Forward-propagation process: calculate all transformation parameters
        if len(matrices) > 1:
            for mat1 in matrices[1:]:
                omega = merge(mat0, mat1, nfm)
                thetaMatrices.append(theta(mat0, mat1, nfm))
                mat0 = omega  # omega connects the initial and final states by the end of this for-loop.
        else:  # only a single interaction is present
            omega = mat0

        finalState = np.dot(omega, initStates)
        # Backward calculation for intermediate states
        tempState = copy.deepcopy(initStates)
        for i, thetaMatrix in enumerate(thetaMatrices[::-1]):
            newState = np.dot(thetaMatrix, tempState)
            states[:, -i - 2] = newState
            if nfm < tnm:
                tempState[nfm:] = newState[nfm:]
            else:
                tempState = newState
        # connect intermediate states with initial and final states
        for i in range(nfm):
            states[i, 0] = initStates[i]
            states[i, -1] = finalState[i]
        for j in range(nfm, tnm):
            states[j, -1] = initStates[j]
            states[j, 0] = finalState[j]
        return states

    def plot(self, initStates, ax1=None, ax2=None):
        """
        Visualize the state evolution and interactions within the edge.

        Args:
            initStates (np.array): Initial state array.
            ax1, ax2 (matplotlib.axes.Axes, optional): Axes on which to plot.

        Returns:
            tuple: Axes with the plots, or False if an error occurs.
        """
        import matplotlib.patches as patches

        tnm = self.totalNumMover
        nfm = self.numForwardMover
        if ax1 is None or ax2 is None:
            _, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax1, ax2 = axs
            plt.subplots_adjust(hspace=0.1)
        seq = self.seq
        try:
            scatterSite = np.arange(0.5, len(seq) + 0.5, 1)
            scatterValue = [scatter[2] for scatter in seq]
            colorsUpper = [
                (
                    [(nfm - scatter[0]) / tnm, 0.1, 0]
                    if scatter[0] < nfm
                    else [0, 0.1, (scatter[0] + 1) / tnm]
                )
                for scatter in seq
            ]
            colorsLower = [
                (
                    [(nfm - scatter[1]) / tnm, 0.1, 0]
                    if scatter[1] < nfm
                    else [0, 0.1, (scatter[1] + 1) / tnm]
                )
                for scatter in seq
            ]

            for xi, yi, colorUpper, colorLower in zip(
                scatterSite, scatterValue, colorsUpper, colorsLower
            ):
                # Define a circle radius and center
                radius = 0.05
                circle_center = (xi, yi)

                # Create upper half (red)
                theta1, theta2 = 0, 180  # Degrees
                upper_half = patches.Wedge(
                    circle_center, radius, theta1, theta2, color=colorUpper
                )

                # Create lower half (blue)
                theta1, theta2 = 180, 360  # Degrees
                lower_half = patches.Wedge(
                    circle_center, radius, theta1, theta2, color=colorLower
                )

                # Add patches to the axes
                ax1.add_patch(upper_half)
                ax1.add_patch(lower_half)

            states = self.status_check(initStates)
            [
                (
                    ax2.plot(states[row, :], color=[(nfm - row) / tnm, 0.1, 0])
                    if row < nfm
                    else ax2.plot(states[row, :], color=[0, 0.1, (row + 1) / tnm])
                )
                for row in range(tnm)
            ]

            ax1.set_ylabel("Value")
            ax1.set_ylim(-0.05, 1.05)
            ax2.set_xlabel("Interaction Site")
            ax2.set_ylabel("Flow Status")

            return (ax1, ax2)
        except Exception as e:
            print(f"Plotting error: {str(e)}")
            return False

    def output_to_json(self, filename="data_edge.json"):
        """Output the edge's transition matrix and sequence information to a JSON file."""

        edgeSequence = self.get_seq()
        edgeMat = self.trans_mat()
        try:
            edge_to_json(filename, edgeSequence, edgeMat)
        except Exception as e:
            print(f"Error: Failed to write to {filename}. {str(e)}")


def system_input_filter(
    nodesCurrent, graph, numForwardMover, totalNumMover, zeroVoltTerminal, blockStates
):
    # Check the data type and structure
    if not isinstance(nodesCurrent, list):
        raise TypeError("Expected nodeCurrent to be a list")
    if not isinstance(graph, list):
        raise TypeError("Expected graph to be a list")
    if not all(isinstance(edge, Edge) for edge in graph):
        raise TypeError("Expected every element in graph to be an instance of Edge")
    if not isinstance(numForwardMover, int):
        raise TypeError("Expected number of forward movers is an non-negative integer")
    if not isinstance(zeroVoltTerminal, int):
        raise TypeError(
            "Expected index of the zero-voltage terminal is an non-negative integer"
        )
    # Check the data value
    if len(graph) is not len(nodesCurrent):
        raise ValueError(
            "The graph size "
            + str(len(graph))
            + " and the provided current of nodes "
            + str(len(nodesCurrent))
            + " do not match. "
        )
    if zeroVoltTerminal >= len(graph):
        raise ValueError(
            "The index of the zero-voltage terminal is out of range from 0 to "
            + str(len(graph) - 1)
        )
    if numForwardMover > totalNumMover:
        raise ValueError(
            "The number of forward movers should be not larger than the total number of movers"
            + str(len(graph))
        )
    if blockStates is not None:
        if not check_blockstates_structure(blockStates):
            raise TypeError(
                "Expected blockstates has a structure formulated as [[# terminal index, [# all blocked states from this terminal],...]]"
            )
        if not check_blockstates_value(
            blockStates, graph[0].trans_mat().shape[0], len(graph)
        ):
            raise ValueError("The blockstates contains unphysical parameters")
    return True


def edge_input_filter(sequence, totalNumMover, numForwardMover):
    # Check the data type and structure
    if not isinstance(sequence, np.ndarray):
        raise TypeError("Expected sequence to be a numpy.ndarray")
    if not isinstance(totalNumMover, int):
        raise TypeError("Expected total number of movers is an non-negative integer")
    if not isinstance(numForwardMover, int):
        raise TypeError("Expected number of forward movers is an non-negative integer")
    if not check_sequence_structure(sequence):
        raise TypeError(
            "Expected sequence has a structure formulated as [[Flow #1(int), Flow #2(int), Value(float), Number(int)],...]"
        )
    # Check the data value
    if numForwardMover > totalNumMover:
        raise ValueError(
            "The number of forward movers should be not larger than the total number of movers "
            + str(totalNumMover)
        )
    if not check_sequence_value(sequence, totalNumMover):
        raise ValueError(
            "The sequence contains unphysical parameters. Check your sequence input."
        )
    return True
