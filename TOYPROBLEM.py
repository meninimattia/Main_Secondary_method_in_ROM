# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr, spsolve
from matplotlib import pyplot as plt
import time

np.random.seed(1)

data = [1, 1000, 2500, 50, 50]

# Number of DoFs
nn = data[1]

# Number of DoFs - Number of secondary DoFs (= internal + main DoFs)
nt = nn-int(2*np.sqrt(nn))

# Creation of the matrices
M = np.identity(nn)

v_1 = np.ones(nn-1)
v_2 = -2*np.ones(nn)
Lap = np.diag(v_1,-1)+np.diag(v_2,0)+np.diag(v_1,1)

# Creation of the MPC matrix
case = data[0]
if case == 1:
	# CASE 1 
	T = np.zeros((nn,nt))
	for i in range(0,nt):
		T[i,i] = 1
	for i in range(0, nn):
		T[i, i-nt] = 1
	np.random.shuffle(T)
else:
	# CASE 2   
	T = np.zeros((nn,nt))
	for i in range(0,nt):
		T[i,i] = 1
	for i in range(nt, nn):
		index = np.random.choice(np.arange(0, nt), size = 3, replace = False)
		T[i,index[0]] = 1/np.random.choice(np.arange(1,10))
		T[i,index[1]] = 1/np.random.choice(np.arange(1,10))
		T[i,index[2]] = 1/np.random.choice(np.arange(1,10))
	np.random.shuffle(T)
T = csr_matrix(T)

# Vector for the MPC application
b = np.ones(nn)
for i in range(0,nn):
    b[i] = i

# Time discretization
nsteps = data[2]
t_0 = 0
t_end = 5
delta_t = (t_end-t_0)/nsteps
print(f"\nThe interval of time is: {delta_t}")

# FOM system
K = M + delta_t*Lap
K = csr_matrix(K)
M = csr_matrix(M)

# Congruential transformation of the system
K_hat = T.T @ K @ T

# Solutions
snapshots = np.zeros((nn,nsteps))
sol_1 = np.zeros((nn,nsteps))
sol_2 = np.zeros((nn,nsteps))
sol_3 = np.zeros((nn,nsteps))
sol_4 = np.zeros((nn,nsteps))
sol_5 = np.zeros((nn,nsteps))

# Initial condition
u_0 = np.zeros(nn)
for i in range(0,nn):
    u_0[i] = i
u = u_0

# FOM solution
ti_FOM = time.time()
for i in range(0,nsteps):
    f = M @ u - K @ b
    f_hat = T.T @ f 
    u_hat = spsolve(K_hat, f_hat)
    u = T @ u_hat + b 
    snapshots[:,i] = u

ControlPlot = False
if ControlPlot == True:
	plt.figure(2)
	print_nodes = np.linspace(0, 10, nn)
	
	ti_FOM = time.time()
	for i in range(0, nsteps):
    	f = M @ u - K @ b
    	f_hat = T.T @ f 
    	u_hat = spsolve(K_hat, f_hat)
    	u = T @ u_hat + b 
    	snapshots[:, i] = u
	
    	plt.cla()         
    	plt.plot(print_nodes, u, 'b-')
    	plt.title(f"FOM Solution - Step {i}")
    	plt.xlabel("Nodes")
    	plt.ylabel("u")
    	plt.pause(0.01)    
	
	plt.show()

time_FOM = time.time() - ti_FOM
print(f"\nThe time to solve the FOM system is: {time_FOM}")

# POD, computation of the SVD
ti_SVD = time.time()
U, S, Vt = linalg.svd(snapshots, full_matrices=False)

# Graph of the singular values
ControlGraph = False
if ControlGraph == True:
    n_data = np.arange(1,len(S) + 1)
    plt.figure(1)
    plt.plot(n_data, S, marker = 'o', linestyle = '-', color = 'm')
    plt.yscale('log')
    plt.xlabel('Number of singular values')
    plt.ylabel('Value of singular values')
    plt.grid(False)
    plt.show()
   
# Computation of the basis for the POD -> since the problem is not physical this is not good
# indeed the first singular value is 2 orders of magnitude higher than the second one -> only one mode
ControlNumberModes = False
norm_S = linalg.norm(snapshots)
if ControlNumberModes == True:
	m = np.size(S)
	k = m
	truncation_tolerance = 1e-12
	for t_1 in range(1,m):
		numerator = 0
		denominator = 0
		for t_2 in range(t_1,m):
			numerator += S[t_2]**2
		for t_3 in range (0,t_1):
			denominator += S[t_3]**2
		if np.sqrt(numerator/denominator) <= (truncation_tolerance * norm_S):
			k = t_1
			break 
    
k = data[3] # To fix a value
 
Phi = U[:,:k]
print(f"\nThe basis shape for X is: {np.shape(Phi)}")
time_SVD = time.time() - ti_SVD
print(f"The time for the computation of the basis is: {time_SVD}")


# FULL SNAPSHOTS matrix
print(f"\nFull Snapshots matrix")
# Four approaches:
#   - QR decomposition
#   - PseudoInverse of T
#   - Least Square solution
#   - Main Rows of the basis

Kb = K @ b # used in more than one approaches 

# Approach 1: QR decomposition
ti_1 = time.time()
Q, R = np.linalg.qr(T.toarray())
m_F1 = Phi.T @ Q @ np.linalg.pinv(R).T @ T.T 
K_1 = m_F1 @ K @ m_F1.T
K_1 = csr_matrix(K_1)
m_U1 = np.linalg.pinv(R) @ Q.T @ Phi
b_U1 = np.linalg.pinv(R) @ Q.T @ b 
b_F1 = m_F1 @ K @ T @ b_U1

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb
	f_1 = m_F1 @ f + b_F1
	q = spsolve(K_1, f_1)
	u_hat = m_U1 @ q - b_U1
	u = T @ u_hat + b
	sol_1[:,i] = u
  
time_sol1 = time.time() - ti_1
norm_1 = linalg.norm(snapshots-sol_1)/norm_S
print(f"\nAPPROACH 1")
print(f"The norm of the error with the FOM simulation is: {norm_1}")
print(f"The time for the ROM simulation is: {time_sol1}")

del sol_1

# Approach 2: PseudoInverse of T
ti_2 = time.time()
m_F2 = Phi.T @ linalg.pinv(T.toarray()).T @ T.T
K_2 = m_F2 @ K @ m_F2.T
K_2 = csr_matrix(K_2)
m_U2 = linalg.pinv(T.toarray()) @ Phi
b_U2 = linalg.pinv(T.toarray()) @ b
b_F2 = m_F2 @ K @ T @ b_U2

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb
	f_2 = m_F2 @ f + b_F2
	q = spsolve(K_2, f_2)
	u_hat = m_U2 @ q - b_U2
	u = T @ u_hat + b
	sol_2[:,i] = u

time_sol2 = time.time() - ti_2
norm_2 = linalg.norm(snapshots - sol_2)/norm_S
print(f"\nAPPROACH 2")
print(f"The norm of the error with the FOM simulation is: {norm_2}")
print(f"The time for the ROM simulation is: {time_sol2}")

del sol_2

# Approach 3: Least Square solution
ti_3 = time.time()
# Least square solution
LS_Phi = np.empty((nt,k))
for i in range(0,k):
	LS_Phi[:,i] = lsqr(T, Phi[:,i])[0]
LS_b = lsqr(T, b)[0]
KTLS_b = K @ T @ LS_b
TLS_Phi = T @ LS_Phi
K_3 = TLS_Phi.T @ K @ TLS_Phi
K_3 = csr_matrix(K_3)

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb + KTLS_b
	f_3 = TLS_Phi.T @ f
	q = spsolve(K_3, f_3)
	u_hat = LS_Phi @ q - LS_b
	u = T @ u_hat + b
	sol_3[:,i] = u

time_sol3 = time.time() - ti_3
norm_3 = linalg.norm(snapshots - sol_3)/norm_S
print(f"\nAPPROACH 3")
print(f"The norm of the error with the FOM simulation is: {norm_3}")
print(f"The time for the ROM simulation is: {time_sol3}")

del sol_3

# Approach 4: Main Rows of the basis
ti_4 = time.time()
MR_Phi = np.empty((nt,k))

# Code used to find the main rows of a matrix
def find_main_rows(matrix):
	vect = [] # Vector of the "main" rows (they correspond to the main DoFs -> in the 1D case DoFs == Nodes)
	columns = []
	mask = np.zeros(matrix.shape[1]) # To control if a particular column of T is already controlled
	# This is done in order to keep only the first 0...0i0...0 type of rows
	row = 0 # To control all the rows of the matrix T
	row_length = matrix.indptr[1] # Initilization of the length of each row (== number of values in that particular row)
	for idx, col in enumerate(matrix.indices): # for cycle over the columns 
		if idx == matrix.indptr[row+1]: # used to change the row (the indptr points to the first value of each row)
			row += 1 # row controlled 
			row_length = matrix.indptr[row+1]-matrix.indptr[row] # Number of entries in the particular row
		# print(f"row {row}, col {col}, val {matrix.data[idx]}")
		if not(mask[col]) and row_length==1: # Storage only of the rows related to the columns never seen and with only one value inside
			vect.append(row) # indices of the row stored
			columns.append(int(col))
			mask[col] = 1.0 # column seen
	order = np.zeros(len(columns), dtype=int)
	for i in range(0, len(columns)):
		order[columns[i]] = i
	return vect, order
rows, order = find_main_rows(T)

MR_Phi = Phi[rows,:]
MR_Phi = MR_Phi[order]
MR_b = b[rows]
MR_b = MR_b[order]
KTMR_b = K @ T @ MR_b
TMR_Phi = T @ MR_Phi
K_4 = TMR_Phi.T @ K @ TMR_Phi
K_4 = csr_matrix(K_4)

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb + KTMR_b
	f_4 = TMR_Phi.T @ f
	q = spsolve(K_4, f_4)
	u_hat = MR_Phi @ q - MR_b
	u = T @ u_hat + b
	sol_4[:,i] = u

time_sol4 = time.time() - ti_4
norm_4 = linalg.norm(snapshots - sol_4)/norm_S
print(f"\nAPPROACH 4")
print(f"The norm of the error with the FOM simulation is: {norm_4}")
print(f"The time for the ROM simulation is: {time_sol4}")

del sol_4

# REDUCED SNAPSHOTS matrix (Only "main rows" of the snapshots mastrix)
print(f"\nReduced Snapshots matrix")

snapshots_t = snapshots[rows, :] - b[rows, np.newaxis]
snapshots_t = snapshots_t[order]

# POD, computation of the SVD
ti_SVD_t = time.time()
U_t, S_t, Vt_t = linalg.svd(snapshots_t, full_matrices=False)

# Graph of the singular values
if ControlGraph == True:
    n_data = np.arange(1,len(S_t) + 1)
    plt.figure()
    plt.plot(n_data, S_t, marker = 'o', linestyle = '-', color = 'm')
    plt.yscale('log')
    plt.xlabel('Number of singular values')
    plt.ylabel('Value of singular values')
    plt.grid(False)
    plt.show
   
# Computation of the basis for the POD
norm_S_t = linalg.norm(snapshots_t)
m = np.size(S_t)
k = m
truncation_tolerance_t = 1e-16
for t_1 in range(1,m):
	numerator = 0
	denominator = 0
	for t_2 in range(t_1,m):
		numerator += S[t_2]**2
	for t_3 in range (0,t_1):
		denominator += S[t_3]**2
	if np.sqrt(numerator/denominator) <= (truncation_tolerance_t * norm_S_t):
		k = t_1
		break 
    
k = data[4] # To fix a value
 
Phi_t = U_t[:,:k]
print(f"\nThe basis shape for X is: {np.shape(Phi_t)}")
time_SVD_t = time.time() - ti_SVD_t
print(f"The time for the computation of the basis is: {time_SVD_t}")

ti_5 = time.time()

TPhi_t = T @ Phi_t
K_5 = TPhi_t.T @ K @ TPhi_t
K_5 = csr_matrix(K_5)

u = u_0
for i in range(0,nsteps):
	f = M @ u - Kb
	f_5 = TPhi_t.T @ f
	q = spsolve(K_5, f_5)
	u = TPhi_t @ q + b
	sol_5[:,i] = u

time_sol5 = time.time() - ti_5
norm_5 = linalg.norm(snapshots - sol_5)/norm_S_t
print(f"\nThe norm of the error with the FOM simulation is: {norm_5}")
print(f"The time for the ROM simulation is: {time_sol5}")

