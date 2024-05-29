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

import os
import random
import json
import numpy as np

__all__ = [
    "generate_bynumber",
    "check_blockstates_structure",
    "check_blockstates_value",
    "check_sequence_structure",
    "check_sequence_value",
    "system_to_json",
    "edge_to_json",
    "NumpyArrayEncoder",
]


def generate_bynumber(messages):
    """Generate a sequence according to given messages.
    A "message" should be formatted into a nested list structure, e.g.,
    [[0,1,0.4,3],[0,2,[0.3,0.1],5],[1,2,0.5,3]]. Inside the first item [0,1,0.4,3], the first
    two items 0 and 1 indicate which two edge states participate, the third item 0.4 gives
    the value(s) for this matrix, the last item 3 represents the number of scattering events.
    """
    seq_in_list = []
    for message in messages:
        seq_in_list.extend([message[:3]] * message[3])
        random.shuffle(seq_in_list)
    return np.array(seq_in_list)


def check_blockstates_structure(blockStates):
    if not isinstance(blockStates, list):
        return False
    for item in blockStates:
        # Check if each item in the data is a list of length 2
        if not (isinstance(item, list) and len(item) == 2):
            return False
        # Check if the first element of each item is an integer
        if not isinstance(item[0], int):
            return False
        # Check if the second element is a list of integers
        if not (isinstance(item[1], list) and all(isinstance(x, int) for x in item[1])):
            return False
    return True


def check_blockstates_value(blockStates, totalNumMover, numTerminal):
    for item in blockStates:
        if item[0] > numTerminal - 1:
            return False
        else:
            for state in item[1]:
                if state > totalNumMover - 1:
                    return False
    return True


def check_sequence_structure(sequence):
    if not isinstance(sequence, np.ndarray):
        return False
    sequence_list = sequence.tolist()
    for item in sequence_list:
        # Check if each item is a list of length 3
        if not (isinstance(item, list) and len(item) == 3):
            return False
        # Check if the first two elements are integers
        if not all(isinstance(x, (int, float)) for x in item[:2]):
            return False
        # Check if the third elements are floats
        if not isinstance(item[2], (int, float)):
            return False
    return True


def check_sequence_value(sequence, totalNumMover):
    sequence_list = sequence.tolist()
    for item in sequence_list:
        if all(state > totalNumMover - 1 for state in item[:2]):
            return False
        if item[2] > 1 or item[2] < 0:
            return False
    return True


def system_to_json(
    file_path,
    edgeSequence,
    edgeMat,
    stateInfo,
    nodesCurrent,
    sysMat,
    termVolts,
    blockStates,
):
    data = {
        "edge information": {
            "edgeSequence": edgeSequence,
            "edgeMatrix": edgeMat,
            "stateInformation": stateInfo,
        },
        "system information": {
            "nodesCurrent": nodesCurrent,
            "systemMatrix": sysMat,
            "terminalVoltages": termVolts,
            "blockStates": blockStates,
        },
    }
    if os.path.exists(file_path):
        # Ask user for action
        response = input(
            f"The file {file_path} already exists. Do you want to overwrite it? (yes/no): "
        ).lower()
        if response != "yes":
            print("Operation cancelled.")
            return
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyArrayEncoder)
    print(f"Data written to {file_path}")


def edge_to_json(file_path, edgeSequence, edgeMat):
    data = {
        "edgeSequence": edgeSequence,
        "edgeMatrix": edgeMat,
    }
    if os.path.exists(file_path):
        # Ask user for action
        response = input(
            f"The file {file_path} already exists. Do you want to overwrite it? (yes/no): "
        ).lower()
        if response != "yes":
            print("Operation cancelled.")
            return
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, cls=NumpyArrayEncoder)
    print(f"Data written to {file_path}")


class NumpyArrayEncoder(json.JSONEncoder):
    """A custom JSON encoder that handles NumPy arrays"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
