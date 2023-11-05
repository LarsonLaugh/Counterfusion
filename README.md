# Interaction Matrix Toolbox

The Interaction Matrix Toolbox is a set of Python functions for working with interaction matrices in the context of a scientific project. These functions allow you to create, manipulate, and visualize interaction matrices, which are used to represent interactions between flows.

## Functions

### 1. `generate_bynumber`

Generates a sequence of interactions containing specific numbers of Alpha, Beta, and Delta interactions. You can specify the number of each type of interaction and whether their strengths should be randomized.

### 2. `generate_byprob`

Generates a sequence of interactions where interactions Alpha, Beta, and Delta appear based on given probabilities. You can specify the length of the sequence, the probabilities for each interaction type, and whether their strengths should be randomized.

### 3. `visualseq`

Visualizes a sequence of interactions using a scatter plot. The x-axis represents the index of each interaction, and the y-axis represents the numerical parameter of each interaction. The color of each point corresponds to the type of interaction (Alpha, Beta, Delta).

### 4. `theta`

Calculates the Theta matrix based on two individual interaction matrices (Om1 and Om2). The Theta matrix represents the combined effect of Om1 and Om2 on the system.

### 5. `merge`

Calculates a new Omega interaction matrix representing the combined effect of two individual interactions (Om1 and Om2). This function uses the Theta matrix calculated by the `theta` function.

### 6. `Beta`, `Alpha`, `Delta`

Functions to create specific interaction matrices: Beta, Alpha, and Delta. These functions allow you to specify the strength of the interactions and enforce valid parameter ranges.

## Usage

1. Import the Interaction Matrix Toolbox functions into your Python script or project.

2. Use the functions to create, manipulate, and visualize interaction matrices according to your project's needs.

Example usage:
```python
# Generate a sequence of interactions by specifying numbers
seq = generate_bynumber(num_alpha=10, num_beta=5, num_delta=7, random_value=True, alpha=[0.2, 0.8], beta=0.6, delta=0.4)

# Generate a sequence of interactions by specifying probabilities
seq = generate_byprob(length=200, prob_alpha=0.3, prob_beta=0.4, random_value=True, alpha=[0.2, 0.8], beta=[0.3, 0.6], delta=[0.1, 0.5])

# Visualize the sequence of interactions
visualseq(seq)

# Calculate Theta matrix
theta_matrix = theta(Om1, Om2)

# Merge two interaction matrices
merged_matrix = merge(Om1, Om2)

# Create specific interaction matrices
beta_matrix = Beta(0.6)
alpha_matrix = Alpha(0.8)
delta_matrix = Delta(0.4)
```
Remember to handle exceptions and validation as described in the individual function documentation.