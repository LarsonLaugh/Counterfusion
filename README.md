# CounterFusion

---

This repository implements algorithms for a paper in preparation. To dive deeper, please visit
[my personal website](http://lixianphwang.com/projects/1_project/).

## Prerequisites

To effectively engage with the notebook, the following Python packages are required:

- `numpy`: For handling numerical computations and array operations.
- `matplotlib`: Essential for creating and displaying graphs and visualizations.

## Installation

Ensure you have the necessary packages installed by running:

```bash
pip install numpy matplotlib
```

## Usage

For details, please refer to the [jupyter notebook](demo.ipynb) which provides many demo cases. Here you may find python
snippets to showcase step-by-step the build-up from the bottom, a single interaction, to a system containing multiple sequences of
interactions.

1. Define a single interaction

```python
from counterfusion import interaction_builder
left_stochastic_matrix = interaction_builder(dim=2,id1=0,id2=1,value=[0.1,0.3])
right_stochastic_matrix = left_stochastsic_matrix.T
doubly_stochastic_matrix = interaction_builder(dim=2,id1=0,id2=1,value=0.5)
```

2. Define a sequence of interactions

```python
from counterfusion import *
#===============================================================
# General information for the edge - Hyperparameters
totalNumMover = 4
numForwardMover = 2
initStates = [1,1,0.2,0.2]
#===============================================================
# Information of scattering events 
# Interaction parameters
v03 = 0.3
v01 = 0.5
v23 = 0.8
edgeDef = [[0,3,v03,10],[0,1,v01,10],[2,3,v23,10]]
edgeInfo = generate_bynumber(edgeDef)
edge = Edge(edgeInfo,totalNumMover,numForwardMover)
```
3. Define a system constituted by several Edge instances.

```python
from counterfusion import *
# Define a six-terminal system
# C1--M1--C2--M2--C3--M3--C4--M4--C5--M5--C6--M6--C1
# Total number of edge states: 4
# Number of forward-moving edge states: 2 (#0,#1)
# Number of backward-moving edge states: 2 (#2,#3)
#===============================================================
# General information for the system - Hyperparameters
totalNumMover = 4
numForwardMover = 2
zeroVoltTerminal = 3
#===============================================================
# Information of scattering events 
# Interaction parameters
v02 = 0.9
v13 = 0.7
v12 = 0.3
# Define interaction between nodes (contacts)
# C1--M1--C2
edgeDef1 = [[0,2,v02,10],[1,3,v13,10]]
# C2--M2--C3
edgeDef2 = [[0,2,v02,10],[1,2,v12,10]]
# C3--M3--C4
edgeDef3 = [[0,2,v02,10],[1,3,v13,10]]
# C4--M4--C5
edgeDef4 = [[0,2,v02,10],[1,2,v12,10]]
# C5--M5--C6
edgeDef5 = [[0,2,v02,10],[1,3,v13,10]]
# C6--M6--C1
edgeDef6 = [[0,2,v02,10],[1,2,v12,10]]
#================================================================
edgesDef = [edgeDef1,edgeDef2,edgeDef3,edgeDef4,edgeDef5,edgeDef6]
edgesInfo = []
for edgeDef in edgesDef:
    edgesInfo.append(generate_bynumber(edgeDef))
graph = []
for edgeInfo in edgesInfo:
    graph.append(Edge(edgeInfo,totalNumMover,numForwardMover))
nodesCurrent = [1,0,0,-1,0,0]
sys = System(nodesCurrent,graph,numForwardMover,zeroVoltTerminal)
```

4. Define individual paths of flows

```python
# The definition of blocking_state should strictly follow this rule: 
# [[index_of_terminal#1,[all blocked states in this terminal]],[[index_of_terminal#2,[all blocked states in this terminal],...]]]
blockStates = [[1,[0]],[0,[2]],[2,[3]],[3,[1]]]
sys = System(nodesCurrent,graph,numForwardMover,zeroVoltTerminal,blockStates)
```

## License
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.