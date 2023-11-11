from scatter import generate_bynumber, fusion
import matplotlib.pyplot as plt
import numpy as np

def calculator(alpha_value,beta_value,delta_value,num_alpha=[1]*6,num_beta=[1]*6,num_delta=[1]*6,I=1):
    # generate transformation matrix between adjacent contacts
    M1 = fusion(generate_bynumber(num_alpha=na1, num_beta=nb1, num_delta=nd1, random_value=False, alpha=alpha_value, beta=beta_value, delta=delta_value))[0]
    M2 = fusion(generate_bynumber(num_alpha=na2, num_beta=nb2, num_delta=nd2, random_value=False, alpha=alpha_value, beta=beta_value, delta=delta_value))[0]
    M3 = fusion(generate_bynumber(num_alpha=na3, num_beta=nb3, num_delta=nd3, random_value=False, alpha=alpha_value, beta=beta_value, delta=delta_value))[0]
    M4 = fusion(generate_bynumber(num_alpha=na4, num_beta=nb4, num_delta=nd4, random_value=False, alpha=alpha_value, beta=beta_value, delta=delta_value))[0]
    M5 = fusion(generate_bynumber(num_alpha=na5, num_beta=nb5, num_delta=nd5, random_value=False, alpha=alpha_value, beta=beta_value, delta=delta_value))[0]
    M6 = fusion(generate_bynumber(num_alpha=na6, num_beta=nb6, num_delta=nd6, random_value=False, alpha=alpha_value, beta=beta_value, delta=delta_value))[0]

    M2 = M2[np.ix_([0,2],[0,2])]
    M3 = M3[np.ix_([0,2],[0,2])]
    # Perform calculations
    M11 = 3 - (M6[0, 2] + M6[1, 2] + M1[2, 0] + M1[2, 1] + M1[2, 2] * M2[1, 0] * (M1[0, 0] + M1[0, 1]) / (1 - M2[1, 0] * M1[0, 2]))
    M12 = -(M2[1, 1] * M1[2, 2]) / (1 - M2[1, 0] * M1[0, 2])
    M13 = 0
    M14 = -(M6[0, 0] + M6[0, 1] + M6[1, 0] + M6[1, 1])

    M21 = -(M2[0, 0] * (M1[0, 0] + M1[0, 1]) * (1 + M1[0, 2] * M2[1, 0] / (1 - M2[1, 0] * M1[0, 2])) + M3[1, 1] * M4[2, 1] * (M1[1, 0] + M1[1, 1]) / (1 - M4[2, 0] * M3[0, 1]) + M3[1, 1] * M4[2, 1] * M1[1, 2] * M2[1, 0] * (M1[0, 0] + M1[0, 1]) / ((1 - M4[2, 0] * M3[0, 1]) * (1 - M2[1, 0] * M1[0, 2])))
    M22 = 2 - (M2[0, 1] + M3[1, 0] + M1[0, 2] * M2[1, 1] * M2[0, 0] / (1 - M2[1, 0] * M1[0, 2]) + M3[1, 1] * M4[2, 0] * M3[0, 0] / (1 - M4[2, 0] * M3[0, 1]) + M3[1, 1] * M4[2, 1] * M1[1, 2] * M2[1, 1] / ((1 - M4[2, 0] * M3[0, 1]) * (1 - M2[1, 0] * M1[0, 2])))
    M23 = -(M4[2, 2] * M3[1, 1]) / (1 - M4[2, 0] * M3[0, 1])
    M24 = 0

    M31 = -((M4[1, 1] + M4[0, 1]) * (M1[1, 0] + M1[1, 1]) + (M4[1, 1] + M4[0, 1]) * M1[1, 2] * M2[1, 0] * (M1[0, 0] + M1[0, 1]) / (1 - M2[1, 0] * M1[0, 2]) + ((M4[0, 0] + M4[1, 0]) * M3[0, 1] * M4[2, 1]) / (1 - M4[2, 0] * M3[0, 1]) * (M1[1, 0] + M1[1, 1] + M2[1, 0] * M1[1, 2] * (M1[0, 0] + M1[0, 1]) / (1 - M2[1, 0] * M1[0, 2])))
    M32 = -((M4[1, 1] * M1[1, 2] * M2[1, 1]) / (1 - M2[1, 0] * M1[0, 2]) + M4[1, 0] * M3[0, 0] + M4[1, 0] * M3[0, 1] * M4[2, 0] * M3[0, 0] / (1 - M4[2, 0] * M3[0, 1]) + M4[1, 0] * M3[0, 1] * M4[2, 1] * M1[1, 2] * M2[1, 1] / ((1 - M4[2, 0] * M3[0, 1]) * (1 - M2[1, 0] * M1[0, 2])) + M4[0, 1] * M1[1, 2] * M2[1, 1] / (1 - M2[1, 0] * M1[0, 2]) + M4[0, 0] * M3[0, 0] + M4[0, 0] * M3[0, 1] * M4[2, 0] * M3[0, 0] / (1 - M4[2, 0] * M3[0, 1]) + M4[0, 0] * M3[0, 1] * M4[2, 1] * M1[1, 2] * M2[1, 1] / ((1 - M4[2, 0] * M3[0, 1]) * (1 - M2[1, 0] * M1[0, 2])))
    M33 = 3 - (M4[0, 2] + M4[1, 2] + M5[2, 0] + M5[2, 1] + M4[1, 0] * M3[0, 1] * M4[2, 2] / (1 - M4[2, 0] * M3[0, 1]) + M4[0, 0] * M3[0, 1] * M4[2, 2] / (1 - M4[2, 0] * M3[0, 1]))
    M34 = -M5[2, 2]

    M41 = -M6[2, 2]
    M42 = 0
    M43 = -(M5[0, 0] + M5[0, 1] + M5[1, 0] + M5[1, 1])
    M44 = 3 - (M5[0, 2] + M5[1, 2] + M6[2, 0] + M6[2, 1])

    # Construct matrix M using the calculated elements
    M = np.array([[M11, M12, M13, M14], [M21, M22, M23, M24], [M31, M32, M33, M34], [M41, M42, M43, M44]])

    # Construct matrix R using specific elements from matrix M
    R = np.array([[M11, M12, M14], [M21, M22, M24], [M31, M32, M34]])

    # Define the vector p
    p = np.array([I, 0, -I])

    # Solve for vector S using linear algebra (Solving the system R * S = p)
    S = np.linalg.solve(R, p)

    # Print the solution vector S
    # print("Solution vector S:", S)

    # Define a dictionary to hold all settings and their values
    settings = {
        "Solution Vector": S,
        "Alpha Value": alpha_value,
        "Beta Value": beta_value,
        "Delta Value": delta_value,
        "Current Magnitude (I)": I,
        "na1": na1, "na2": na2, "na3": na3, "na4": na4, "na5": na5, "na6": na6,
        "nb1": nb1, "nb2": nb2, "nb3": nb3, "nb4": nb4, "nb5": nb5, "nb6": nb6,
        "nd1": nd1, "nd2": nd2, "nd3": nd3, "nd4": nd4, "nd5": nd5, "nd6": nd6
    }
    return S, settings


# Interaction settings between adjacent contacts
# Define your own settings here
# Initialize numbers of alpha, beta, and delta contacts
alpha_value = 0
beta_value = 0.2
# delta_value = 0.9999

na1 = na2 = na3 = na4 = na5 = na6 = 0

nb1 = nb4 = 8
nb5 = nb6 = 16
nb2 = nb3 = 0

nd1 = nd4 = 8
nd5 = nd6 = 16
nd2 = nd3 = 8

na = [na1,na2,na3,na4,na5,na6]
nb = [nb1,nb2,nb3,nb4,nb5,nb6]
nd = [nd1,nd2,nd3,nd4,nd5,nd6]
V1, V2, V4 = [], [], []
# beta_list = np.linspace(0,1,1000)
delta_list = np.linspace(0,1,1000)

for delta_value in delta_list:
    mu, settings = calculator(alpha_value=0,beta_value=beta_value,delta_value=delta_value,num_alpha=[1]*6,num_beta=[1]*6,num_delta=[1]*6)
    V1.append(mu[0])
    V2.append(mu[1])
    V4.append(mu[2])

# Save results and settings to a text file
# Open the file and write settings and solution
with open("results_and_settings.txt", "w") as file:
    # Write setting parameters
    file.write("Settings:\n")
    for key, value in settings.items():
        file.write(f"{key}: {value}\n")
    # Write a separator for clarity
    file.write("\n")

# plt.scatter(beta_list,V2,s=10)
# plt.plot(beta_list,V2)
# plt.scatter(beta_list,V1,s=10)
# plt.plot(beta_list,V1)
# plt.scatter(beta_list,V4,s=10)
# plt.plot(beta_list,V4)
plt.scatter(delta_list,V2,s=10)
plt.plot(delta_list,V2)
plt.scatter(delta_list,V1,s=10)
plt.plot(delta_list,V1)
plt.scatter(delta_list,V4,s=10)
plt.plot(delta_list,V4)
plt.show()

# Test data
# M1 = np.array([[0.28820226, 0.27720269, 0.43459505],
#                [0.2178337, 0.21969339, 0.56247291],
#                [0.49396403, 0.50310392, 0.00293204]])
#
# M4 = np.array([[0.53735814, 0.17277425, 0.28986761],
#                [0.2151337, 0.08119532, 0.70367098],
#                [0.24750816, 0.74603043, 0.0064614]])
#
# M2 = np.array([[0.14465409, 0.85534591],
#                [0.85534591, 0.14465409]])
#
# M3 = np.array([[0.14465409, 0.85534591],
#                [0.85534591, 0.14465409]])
#
# M6 = np.array([[0.353672865, 0.207694992, 0.438632143],
#                [0.276344362, 0.162290015, 0.561365623],
#                [0.369982773, 0.630014993, 0.00000223431346]])
#
# M5 = np.array([[0.188417214, 0.129727309, 0.681855477],
#                [0.403811544, 0.27804748, 0.318140976],
#                [0.407771243, 0.592225211, 0.00000354641924]])