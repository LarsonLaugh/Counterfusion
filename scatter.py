import numpy as np
import matplotlib.pyplot as plt

# =======================================================================================================
# Warning: modifying these functions may lead to incorrect results!
# core functions
def theta(Om1, Om2):
    """
    Calculate the Theta matrix based on individual interaction matrices Om1 and Om2.

    :param Om1: The individual interaction matrix Om1, which should be a 3x3 NumPy array.
    :param Om2: The individual interaction matrix Om2, which should be a 3x3 NumPy array.

    :return: The Theta matrix, a 3x3 NumPy array.
    """
    # Check if Om1 and Om2 are NumPy arrays.
    if type(Om1) != np.ndarray or type(Om2) != np.ndarray:
        raise TypeError("Input should be np.array type")

    # Check if Om1 and Om2 have the correct shape (3x3).
    if Om1.shape != (3, 3) or Om2.shape != (3, 3):
        raise ValueError("Input should be 3-by-3 array")

    # Calculate the elements of the Theta matrix using provided formulas.
    t11 = Om1[0, 0] + Om1[0, 2] * (Om2[2, 0] * Om1[0, 0] + Om2[2, 1] * Om1[1, 0]) / (
                1 - Om2[2, 0] * Om1[0, 2] - Om2[2, 1] * Om1[1, 2])
    t12 = Om1[0, 1] + Om1[0, 2] * (Om2[2, 0] * Om1[0, 1] + Om2[2, 1] * Om1[1, 1]) / (
                1 - Om2[2, 0] * Om1[0, 2] - Om2[2, 1] * Om1[1, 2])
    t13 = Om1[0, 2] * Om2[2, 2] / (1 - Om2[2, 0] * Om1[0, 2] - Om2[2, 1] * Om1[1, 2])
    t21 = Om1[1, 0] + Om1[1, 2] * (Om2[2, 0] * Om1[0, 0] + Om2[2, 1] * Om1[1, 0]) / (
                1 - Om2[2, 0] * Om1[0, 2] - Om2[2, 1] * Om1[1, 2])
    t22 = Om1[1, 1] + Om1[1, 2] * (Om2[2, 0] * Om1[0, 1] + Om2[2, 1] * Om1[1, 1]) / (
                1 - Om2[2, 0] * Om1[0, 2] - Om2[2, 1] * Om1[1, 2])
    t23 = Om1[1, 2] * Om2[2, 2] / (1 - Om2[2, 0] * Om1[0, 2] - Om2[2, 1] * Om1[1, 2])
    t31 = (Om2[2, 0] * Om1[0, 0] + Om2[2, 1] * Om1[1, 0]) / (1 - Om2[2, 0] * Om1[0, 2] - Om2[2, 1] * Om1[1, 2])
    t32 = (Om2[2, 0] * Om1[0, 1] + Om2[2, 1] * Om1[1, 1]) / (1 - Om2[2, 0] * Om1[0, 2] - Om2[2, 1] * Om1[1, 2])
    t33 = Om2[2, 2] / (1 - Om2[2, 0] * Om1[0, 2] - Om2[2, 1] * Om1[1, 2])

    # Create and return the Theta matrix as a 3x3 NumPy array.
    return np.array([[t11, t12, t13], [t21, t22, t23], [t31, t32, t33]])


def merge(Om1, Om2):
    """
    Calculate a new Omega interaction representing the combined effect of Om1 and Om2.

    :param Om1: The individual interaction matrix Omega1, which should be a 3x3 NumPy array.
    :param Om2: The individual interaction matrix Omega2, which should be a 3x3 NumPy array.

    :return: A new Omega interaction as a 3x3 NumPy array.
    """

    # Check if Om1 and Om2 are NumPy arrays.
    if type(Om1) != np.ndarray or type(Om2) != np.ndarray:
        raise TypeError("Input should be np.array type")

    # Check if Om1 and Om2 have the correct shape (3x3).
    if Om1.shape != (3, 3) or Om2.shape != (3, 3):
        raise ValueError("Input should be 3-by-3 array")

    # Calculate the Theta matrix for the given Om1 and Om2.
    th = theta(Om1, Om2)

    # Calculate the elements of the new Omega interaction using provided formulas.
    m11 = Om2[0, 0] * th[0, 0] + Om2[0, 1] * th[1, 0]
    m12 = Om2[0, 0] * th[0, 1] + Om2[0, 1] * th[1, 1]
    m13 = Om2[0, 0] * th[0, 2] + Om2[0, 1] * th[1, 2] + Om2[0, 2]
    m21 = Om2[1, 0] * th[0, 0] + Om2[1, 1] * th[1, 0]
    m22 = Om2[1, 0] * th[0, 1] + Om2[1, 1] * th[1, 1]
    m23 = Om2[1, 0] * th[0, 2] + Om2[1, 1] * th[1, 2] + Om2[1, 2]
    m31 = Om1[2, 0] + Om1[2, 2] * th[2, 0]
    m32 = Om1[2, 1] + Om1[2, 2] * th[2, 1]
    m33 = Om1[2, 2] * th[2, 2]

    # Create and return the new Omega interaction as a 3x3 NumPy array.
    return np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])


def Beta(beta: (float, int)):
    """
    Create a Beta interaction matrix based on the provided value of beta.

    :param beta: A numerical value between 0 and 1 to determine the strength of the Beta interaction.

    :return: The Beta interaction matrix as a 3x3 NumPy array.
    """
    # Check if the input 'beta' is of a valid type and within the allowed range.
    if type(beta) not in [float, int]:
        raise TypeError("Unsupported argument type")
    if beta < 0 or beta > 1:
        raise ValueError("Beta takes a value between 0 and 1")
    else:
        return np.array([[1, 0, 0], [0, 1 - beta / 2, beta / 2], [0, beta / 2, 1 - beta / 2]])


def Alpha(alpha: (float, int)):
    """
    Create an Alpha interaction matrix based on the provided value of alpha.

    :param alpha: A numerical value between 0 and 1 to determine the strength of the Alpha interaction.

    :return: The Alpha interaction matrix as a 3x3 NumPy array.
    """
    # Check if the input 'alpha' is of a valid type and within the allowed range.
    if type(alpha) not in [float, int]:
        raise TypeError("Unsupported argument type")
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha takes a value between 0 and 1")
    else:
        return np.array([[1 - alpha / 2, alpha / 2, 0], [alpha / 2, 1 - alpha / 2, 0], [0, 0, 1]])


def Delta(delta: (float, int)):
    """
    Create a Delta interaction matrix based on the provided value of delta.

    :param delta: A numerical value between 0 and 1 to determine the strength of the Delta interaction.

    :return: The Delta interaction matrix as a 3x3 NumPy array.
    """
    # Check if the input 'delta' is of a valid type and within the allowed range.
    if type(delta) not in [float, int]:
        raise TypeError("Unsupported argument type")
    if delta < 0 or delta > 1:
        raise ValueError("Delta takes a value between 0 and 1")
    return np.array([[1 - delta / 2, 0, delta / 2], [0, 1, 0], [delta / 2, 0, 1 - delta / 2]])
# =======================================================================================================

# helper functions
def generate_byprob(length, prob_alpha, prob_beta, random_value=False, alpha=0.5, beta=0.5, delta=0.5):
    """Generate a sequence of interactions where interactions Alpha, Beta, and Delta appear based on given probabilities.
        :param length: The total number of interactions to be included in this sequence.
        :param prob_Alpha: The probability for interaction Alpha to appear in the sequence.
        :param prob_Beta: The probability for interaction Beta to appear in the sequence.
        :param random_value: A boolean indicating whether interaction strengths should be randomized within defined ranges.
        :param alpha: If random_value is False, specify the strength of interaction Alpha. If random_value is True, provide upper and lower bounds for Alpha interaction strength.
        :param beta: If random_value is False, specify the strength of interaction Beta. If random_value is True, provide upper and lower bounds for Beta interaction strength.
        :param delta: If random_value is False, specify the strength of interaction Delta. If random_value is True, provide upper and lower bounds for Delta interaction strength.

        :return: A sequence of interactions, including their types (Alpha, Beta, Delta) and values of strength.
    """
    if length < 100:
        raise ValueError("Total number of scatters needs to be at least 100 or above to be statistically correct.")
    prob_delta = 1 - prob_alpha - prob_beta
    rng = np.random.default_rng()
    if random_value is False:
        if isinstance(alpha, (int, float)) or isinstance(beta, (int, float)) or isinstance(delta, (int, float)):
            seq = rng.choice([['A', alpha], ['B', beta], ['D', delta]], size=length,
                             p=[prob_alpha, prob_beta, prob_delta])
        else:
            raise ValueError("alpha, beta, delta should be either integer or float as random_value==False")
    else:
        if isinstance(alpha, list) and isinstance(beta, list) and isinstance(delta, list) and len(alpha) == len(
                beta) == len(delta) == 2:
            seq_type = rng.choice(['A', 'B', 'D'], size=length, p=[prob_alpha, prob_beta, prob_delta])
            seq = np.array([['A', alpha[0] + (alpha[1] - alpha[0]) * rng.random()] if t == 'A'
                            else ['B', beta[0] + (beta[1] - beta[0]) * rng.random()] if t == 'B'
            else ['D', delta[0] + (delta[1] - delta[0]) * rng.random()] for t in seq_type])
        else:
            raise ValueError("alpha, beta, delta should be a list with a length of 2 if random_value==True")
    return seq

def generate_bynumber(num_alpha, num_beta, num_delta, random_value=False, alpha=0.5, beta=0.5, delta=0.5):
    """Generate a sequence of interactions containing specific numbers of Alpha, Beta, and Delta interactions.
        :param num_alpha: The desired number of Alpha interactions in this sequence.
        :param num_beta: The desired number of Beta interactions in this sequence.
        :param num_delta: The desired number of Delta interactions in this sequence.
        :param random_value: A boolean indicating whether interaction strengths should be randomized within defined ranges.
        :param alpha: If random_value is False, specify the strength of Alpha interactions. If random_value is True, provide upper and lower bounds for Alpha interaction strength.
        :param beta: If random_value is False, specify the strength of Beta interactions. If random_value is True, provide upper and lower bounds for Beta interaction strength.
        :param delta: If random_value is False, specify the strength of Delta interactions. If random_value is True, provide upper and lower bounds for Delta interaction strength.
    
        :return: A sequence of interactions, including their types (Alpha, Beta, Delta) and values of strength.
    """
    tot_num = num_alpha + num_beta + num_delta

    # Create an empty sequence with placeholders for Alpha, Beta, and Delta interactions.
    seq = np.array([('A', 0.1)] * tot_num)

    if random_value is False:
        # If random_value is False, set interaction strengths directly.
        if isinstance(alpha, (int, float)) or isinstance(beta, (int, float)) or isinstance(delta, (int, float)):
            if num_alpha != 0:
                seq[:num_alpha] = [('A', alpha)] * num_alpha
            if num_beta != 0:
                seq[num_alpha:num_alpha + num_beta] = [('B', beta)] * num_beta
            if num_delta !=0:
                seq[tot_num - num_delta:tot_num] = [('D', delta)] * num_delta
            np.random.shuffle(seq)
        else:
            raise ValueError("alpha, beta, delta should be either integer or float if random_value==False")
    else:
        # If random_value is True, generate random interaction strengths within specified bounds.
        if isinstance(alpha, list) and isinstance(beta, list) and isinstance(delta, list) and len(alpha) == len(
                beta) == len(delta) == 2:
            rng = np.random.default_rng()
            if num_alpha != 0:
                seq[:num_alpha] = [('A', alpha[0] + (alpha[1] - alpha[0]) * rng.random()) for i in range(num_alpha)]
            seq[num_alpha:num_alpha + num_beta] = [('B', beta[0] + (beta[1] - beta[0]) * rng.random()) for i in
                                                   range(num_beta)]
            seq[tot_num - num_delta:tot_num] = [('D', delta[0] + (delta[1] - delta[0]) * rng.random()) for i in
                                                range(num_delta)]
            np.random.shuffle(seq)
        else:
            raise ValueError("alpha, beta, delta should be a list with a length of 2 if random_value==True")
    return seq

def fusion(seq):
    # Create an empty list to store matrices that will be constructed based on the input sequence 'seq'.
    matrices = []

    # Iterate through each element (interaction type 't' and strength 'v') in 'seq'.
    for t, v in zip(seq[:, 0], seq[:, 1]):
        if t == 'A':
            # Create an Alpha matrix with the given strength 'v' and append it to the 'matrices' list.
            matrices.append(Alpha(float(v)))
        elif t == 'B':
            # Create a Beta matrix with the given strength 'v' and append it to the 'matrices' list.
            matrices.append(Beta(float(v)))
        else:
            # Create a Delta matrix with the given strength 'v' and append it to the 'matrices' list.
            matrices.append(Delta(float(v)))

    # Initialize the result matrix 'mat0' with the first matrix in the 'matrices' list.
    mat0 = matrices[0]

    # Merge the matrices in 'matrices' list sequentially and update 'mat0' with the result.
    for mat1 in matrices[1:]:
        res = merge(mat0, mat1)
        mat0 = res

    # Return the final merged matrix 'res' and the input sequence 'seq'.
    return res, seq


def visualseq(seq):
    """
    Visualize a sequence of interactions using a scatter plot.

    :param seq: A sequence of interactions, where each interaction includes its type ('A', 'B', 'D') and a numerical parameter.
    """

    # Create a scatter plot to visualize the interactions.
    # The x-axis represents the index of each interaction, and the y-axis represents the numerical parameter of each interaction.
    # The color of each point on the scatter plot corresponds to the type of interaction: 'A' in green, 'B' in red, 'D' in blue.
    plt.scatter(np.arange(0, len(seq[:, 0]), 1), [float(v) for v in seq[:, 1]],
                c=[[0, 1, 0] if t == 'A' else [1, 0, 0] if t == 'B' else [0, 0, 1] for t in seq[:, 0]], s=1)

    plt.xlabel('# of scattering events')
    plt.ylabel('Scattering parameter')
    plt.show()