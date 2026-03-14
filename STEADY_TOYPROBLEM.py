import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve, lsqr
import time

np.random.seed(1)

nn = 50
N = nn * nn
Domain = 1.0
n_test = int(3/2 * N)
mu = np.linspace(0, 70, n_test)
h = Domain / (nn - 1)

v_1 = -4 * np.ones(N)
v_2 = np.ones(N - 1)
v_2[np.arange(1, N) % nn == 0] = 0
v_3 = np.ones(N - nn)
K = (diags([v_1, v_2, v_2, v_3, v_3], [0, -1, 1, -nn, nn], shape=(N, N)) / h**2).tocsr()

x = np.linspace(0, Domain, nn)
y = np.linspace(0, Domain, nn)
X, Y = np.meshgrid(x, y)

Nt = N - int(2*np.sqrt(N))

case = 1
if case == 1:
    # CASE 1 
    T = np.zeros((N,Nt))
    for i in range(Nt):
        T[i,i] = 1
    for i in range(Nt, N):
        T[i, i-Nt] = 1
    np.random.shuffle(T)
else:
# CASE 2 
    T = np.zeros((N,Nt))
    for i in range(0,Nt):
        T[i,i] = 1
    for i in range(Nt, N):
        index = np.random.choice(np.arange(0, Nt), size = 3, replace = False)
        T[i,index[0]] = 1/np.random.choice(np.arange(1,10))
        T[i,index[1]] = 1/np.random.choice(np.arange(1,10))
        T[i,index[2]] = 1/np.random.choice(np.arange(1,10))
    np.random.shuffle(T)
T = csr_matrix(T)

b = np.ones(N)

snapshots = np.zeros((N, n_test))
sol_1 = np.zeros((N, n_test))
sol_2 = np.zeros((N, n_test))
sol_3 = np.zeros((N, n_test))
sol_4 = np.zeros((N, n_test))
sol_5 = np.zeros((N, n_test))

ti_FOM = time.time()
for i in range(0, n_test):
    a = 1 + 0.5*np.sin(mu[i]*X*Y)
    A = diags(a.reshape(N))
    K_mu = A @ K
    P = T.T @ K_mu @ T
    f = -100*np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = f.reshape(N)
    f_hat = T.T @ (f - K_mu @ b)
    u_t = spsolve(P, f_hat)
    u = T @ u_t + b
    snapshots[:, i] = u
    u = u.reshape((nn, nn))
time_FOM = time.time() - ti_FOM
print(f"\nThe time to solve the FOM system is: {time_FOM}")

plt.imshow(u, extent=[0,1,0,1], origin="lower")
plt.colorbar()
plt.title("Poisson Solution")
plt.show()
# POD, computation of the SVD
ti_SVD = time.time()
U, S, Vt = linalg.svd(snapshots, full_matrices=False)

norm_S = linalg.norm(snapshots)

ControlNumberModes = True
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

Phi = U[:,:k]
print(f"\nThe basis shape for X is: {np.shape(Phi)}")
time_SVD = time.time() - ti_SVD
print(f"The time for the computation of the basis is: {time_SVD}")

# Approach 1: QR decomposition
ti_1 = time.time()
Q, R = np.linalg.qr(T.toarray())
Rinv = np.linalg.pinv(R)
m_F1 = Phi.T @ Q @ Rinv.T @ T.T 
m_U1 = Rinv @ Q.T @ Phi
b_U1 = Rinv @ Q.T @ b 

for i in range(0,n_test):
    a = 1 + 0.5*np.sin(mu[i]*X*Y)
    A = diags(a.reshape(N))
    K_mu = A @ K
    Kb = K_mu @ b

    K_1 = m_F1 @ K_mu @ m_F1.T
    K_1 = csr_matrix(K_1)
    b_F1 = m_F1 @ K_mu @ T @ b_U1

    f = -100*np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = f.reshape(N) - Kb
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
Tinv = linalg.pinv(T.toarray())
m_F2 = Phi.T @ Tinv.T @ T.T
m_U2 = Tinv @ Phi
b_U2 = Tinv @ b

for i in range(0,n_test):
    a = 1 + 0.5*np.sin(mu[i]*X*Y)
    A = diags(a.reshape(N))
    K_mu = A @ K
    Kb = K_mu @ b

    K_2 = m_F2 @ K_mu @ m_F2.T
    K_2 = csr_matrix(K_2)
    b_F2 = m_F2 @ K_mu @ T @ b_U2

    f = -100*np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = f.reshape(N) - Kb
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
LS_Phi = np.empty((Nt,k))
for i in range(0,k):
	LS_Phi[:,i] = lsqr(T, Phi[:,i])[0]
LS_b = lsqr(T, b)[0]
TLS_Phi = T @ LS_Phi
K_3 = TLS_Phi.T @ K @ TLS_Phi
K_3 = csr_matrix(K_3)

for i in range(0,n_test):
    a = 1 + 0.5*np.sin(mu[i]*X*Y)
    A = diags(a.reshape(N))
    K_mu = A @ K
    Kb = K_mu @ b

    KTLS_b = K_mu @ T @ LS_b
    K_3 = TLS_Phi.T @ K_mu @ TLS_Phi
    K_3 = csr_matrix(K_3)

    f = -100*np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = f.reshape(N) - Kb + KTLS_b
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
MR_Phi = np.empty((Nt,k))

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
TMR_Phi = T @ MR_Phi

for i in range(0,n_test):
    a = 1 + 0.5*np.sin(mu[i]*X*Y)
    A = diags(a.reshape(N))
    K_mu = A @ K
    Kb = K_mu @ b

    KTMR_b = K_mu @ T @ MR_b
    K_4 = TMR_Phi.T @ K_mu @ TMR_Phi
    K_4 = csr_matrix(K_4)

    f = -100*np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = f.reshape(N) - Kb + KTMR_b
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
   
# Computation of the basis for the POD
norm_S_t = linalg.norm(snapshots_t)

m = np.size(S_t)
k = m
truncation_tolerance_t = 1e-12
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
 
Phi_t = U_t[:,:k]
print(f"\nThe basis shape for X is: {np.shape(Phi_t)}")
time_SVD_t = time.time() - ti_SVD_t
print(f"The time for the computation of the basis is: {time_SVD_t}")

ti_5 = time.time()

TPhi_t = T @ Phi_t

for i in range(0,n_test):
    a = 1 + 0.5*np.sin(mu[i]*X*Y)
    A = diags(a.reshape(N))
    K_mu = A @ K
    Kb = K_mu @ b

    K_5 = TPhi_t.T @ K_mu @ TPhi_t
    K_5 = csr_matrix(K_5)

    f = -100*np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = f.reshape(N) - Kb
    f_5 = TPhi_t.T @ f
    q = spsolve(K_5, f_5)
    u = TPhi_t @ q + b
    sol_5[:,i] = u

time_sol5 = time.time() - ti_5
norm_5 = linalg.norm(snapshots - sol_5)/norm_S
print(f"\nThe norm of the error with the FOM simulation is: {norm_5}")
print(f"The time for the ROM simulation is: {time_sol5}")