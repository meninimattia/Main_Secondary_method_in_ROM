# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

nn = ... # number of nodes 
nt = ... # number of time step 
t0 = ... # time step with the first instability

matrix = np.empty((nn,nt-t0+1))

row = -1

v_dir = 1

with open('File', 'r') as infile:
    for line in infile:
        line = line.strip()
        if line:
            try:
                numbers = [float(x) for x in line.split()]
                row += 1
                if row >= nn*t0:
                    a = int(numbers[0])-1
                    b = int(np.floor(row/nn)-t0)
                    matrix[a,b] = numbers[v_dir]
            except ValueError:
                pass
            
a = matrix.max()
print(a)

U, S, Vt = np.linalg.svd(matrix, full_matrices = False)

norm_S = np.linalg.norm(matrix, 'fro') # Frobenius norm of the snapshots matrix
print(norm_S)

m = np.size(S)
n = m

accuracies = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

for t in range(0, len(accuracies)):
	accuracy = accuracies[t]
	for t1 in range(1,m):
		numerator = 0
		denominator = 0
		for t2 in range(t1,m):
			numerator += S[t2]**2
		for t3 in range (0,t1):
			denominator += S[t3]**2
		if np.sqrt(numerator/denominator) <= (accuracy * norm_S):
			n = t1
			print('For accuracy = ', accuracy, 'the number of modes requested is', n)
			break

number = np.arange(1, len(S) + 1)

plt.figure(figsize=(10, 6))
plt.plot(number, S, marker='o', linestyle='-', color='k', linewidth=3) 
plt.yscale('log')
if v_dir == 1:
	plt.title('Singular Values Decay, Velocity in x-direction', fontsize=26, fontweight='bold')
else:
	plt.title('Singular Values Decay, Velocity in y-direction', fontsize=26, fontweight='bold')
plt.xlabel('Nodes', fontsize=20)
plt.ylabel('Singular Values', fontsize=20)
plt.grid(False)
plt.show()
