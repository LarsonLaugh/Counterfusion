from allscatter import *
m, mf = 3,2
beta = 0.5
message1 = [[0,1,beta*0.95,2]]
message2 = [[0,1,beta,30]]
message3 = [[0,1,beta*0.95,2]]
message4 = [[0,1,beta,2]]
message5 = [[0,1,beta,30]]
message6 = [[0,1,beta,2]]

messages = [message1,message2,message3,message4,message5,message6]
seqs = []
for message in messages:
    seqs.append(generate_bynumber(message))

matrices = []
for seq in seqs:
    matrices.append(effective_matrix(seq,m,mf))

current = [1,0,0,-1,0,0]
sys = system(current,matrices)
states = sys.voltage_tracker(mf)