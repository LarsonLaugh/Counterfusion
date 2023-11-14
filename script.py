from scatter import *
import matplotlib.pyplot as plt

init_state = [0.5,1,0.3]
seq =  generate_bynumber(50, 50, 50, random_value=False, alpha=0.2, beta=0.8, delta=0.1)
states, _ = states_check(seq,init_state)
v1, v2, v3 = states[0,:],states[1,:],states[2,:]
plt.plot(v1,color='y')
plt.plot(v2,color='g')
plt.plot(v3,color='m')
plt.scatter(np.arange(0.5, len(seq[:, 0])+0.5, 1), [float(v) for v in seq[:, 1]],
            c=[[1, 0, 0] if t == 'A' else [0, 1, 0] if t == 'B' else [0, 0, 1] for t in seq[:, 0]], s=10)
[plt.axvline(x=pos+0.5,color=c,linestyle='-.') for pos,c in zip(np.arange(0, len(seq[:, 0]), 1),[[0, 1, 0] if t == 'A' else [1, 0, 0] if t == 'B' else [0, 0, 1] for t in seq[:, 0]])]
plt.show()