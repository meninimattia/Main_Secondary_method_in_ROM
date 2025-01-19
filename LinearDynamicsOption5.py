# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr, spsolve

# FULL ORDER MODEL

class Domain():
    #1D domain class
    def __init__(self, number_of_elements):
        self.number_of_elements = number_of_elements
        self.SetUp()


    def SetUp(self):
        self.number_of_nodes = self.number_of_elements+1
        self.nodes_coordinates = np.linspace(0,1,self.number_of_nodes)
        self.connectivity = np.c_[np.linspace(0,self.number_of_nodes-2,self.number_of_nodes-1), np.linspace(1,self.number_of_nodes-1,self.number_of_nodes-1) ]


def setup_domain(number_of_elements):
    domain_object = Domain(number_of_elements)
    return domain_object

class ReferenceElement():

    def __init__(self, degree):
        self.degree = degree
        self.SetUp()

    def SetUp(self):
        if self.degree == 1:
            self.reference_domain = np.array([-1,1])
            self.number_nodes = 2
            self.number_gauss_points = 2
            self.gauss_points_locations = np.array([-1/np.sqrt(3) , 1/np.sqrt(3)])
            self.gauss_points_weights = np.array([1,1])
            self.shape_functions_at_gauss_points =  np.array([(1-self.gauss_points_locations)/2  , (1+self.gauss_points_locations)/2])
            self.shape_functions_derivatives_at_gauss_points  =  np.array ([[-0.5, 0.5 ],[-0.5, 0.5]])
            self.shape_functions_second_derivatives_at_gauss_points = np.array ([[0, 0 ],[0, 0]])
        else:
            raise NameError(f'reference element of the selected degree {self.degree} not implemented')

def setup_reference_element(degree):
    reference_element_object = ReferenceElement(degree)
    return reference_element_object

class FOM_simulation(object):

    def __init__( self, number_of_elements=5, total_time=20,time_step_size=.1, element_degree=1):
        self.total_time = total_time
        self.number_of_time_steps =int(total_time/time_step_size)
        self.time_step_size = time_step_size
        self.setup_domain(number_of_elements)
        self.reference_element = setup_reference_element(element_degree)


    def setup_domain(self, number_of_elements):
        self.domain = Domain(number_of_elements)


    def Run(self):
        self.ComputeSystemMatrix()
        self.Solve()


    def ComputeSystemMatrix(self):
        self.K = np.zeros((self.domain.number_of_nodes, self.domain.number_of_nodes))
        self.M = self.K.copy()
        #element loop
        for ith_element in range(self.domain.number_of_elements):
            element_connectivity = self.domain.connectivity[ith_element, :]
            element_coordinates = self.domain.nodes_coordinates[element_connectivity.astype(int)]
            element_length =  element_coordinates[-1] - element_coordinates[0]
            K_element = np.zeros((self.reference_element.number_nodes,self.reference_element.number_nodes))
            M_element = K_element.copy()
            #gauss points loop
            for i in range(self.reference_element.number_gauss_points):
                shape_function_i = self.reference_element.shape_functions_at_gauss_points[i,:]
                shape_function_derivative_i = self.reference_element.shape_functions_derivatives_at_gauss_points[i,:] * 2/element_length
                weight_i = self.reference_element.gauss_points_weights[i] * element_length/2
                K_element += weight_i * (shape_function_derivative_i.reshape(-1,1) @ shape_function_derivative_i.reshape(1,-1))
                M_element += weight_i * (shape_function_i.reshape(-1,1) @ shape_function_i.reshape(1,-1))
            #assembly
            for e_i, i in zip([0,1], element_connectivity.astype(int)):
                for e_j, j in zip([0,1],element_connectivity.astype(int)):
                    self.M[i,j] +=  M_element[e_i,e_j]
                    self.K[i,j] +=  K_element[e_i,e_j]
        self.T = self.MCPmatrix()
        self.K1 = self.T.T @ self.K @ self.T
        self.M1 = self.T.T @ self.M @ self.T

    def MCPmatrix(self):
        t = int(0.90*self.domain.number_of_nodes)
        T = np.zeros((self.domain.number_of_nodes, t))
        T[0,0] = 1
        T1 = np.zeros((self.domain.number_of_nodes-1, t-1))
        for i in range(0,t-1):
            T1[i,i] = 1
        for i in range(t-1, self.domain.number_of_nodes-1):
            T1[i,t-i-1] = 1
        # np.random.shuffle(T1)
        T[1:,1:] = T1
        return T

    def GetInitialDisplacement(self, applied_force):
        applied_force_vector = np.zeros((self.domain.number_of_nodes))
        applied_force_vector[-1] = applied_force
        applied_force_vector = self.T.T @ applied_force_vector
        Displacement_old = np.zeros((self.domain.number_of_nodes))
        Displacement_old[1:] = self.T[1:,1:] @ np.squeeze(np.linalg.solve(self.K1[1:,1:], applied_force_vector[1:]))

        return Displacement_old


    def Solve(self):
        self.set_up_Newmark_coefficients()

        applied_force = 1
        Displacement_old = self.GetInitialDisplacement(applied_force) #solve static problem to determine original deformation state

        Displacement_new = Displacement_old.copy()
        Velocity_old = np.zeros((self.domain.number_of_nodes))
        Acceleration_old = np.zeros((self.domain.number_of_nodes))

        self.SnapshotsMatrixDisplacements = np.zeros((np.shape(Displacement_old)[0],self.number_of_time_steps))
        self.SnapshotsMatrixDisplacements[:,0] = Displacement_old
        self.SnapshotsMatrixVelocities = np.zeros((np.shape(Displacement_old)[0],self.number_of_time_steps))
        self.SnapshotsMatrixVelocities[:,0] = Velocity_old
        self.SnapshotsMatrixAccelerations = np.zeros((np.shape(Displacement_old)[0],self.number_of_time_steps))
        self.SnapshotsMatrixAccelerations[:,0] = Acceleration_old

        F = np.zeros((self.domain.number_of_nodes))
        K_hat = self.K1 + self.a0*self.M1

        for i in range(1,self.number_of_time_steps):
            F_hat = F + self.M @ (self.a0*Displacement_old  +  self.a2*Velocity_old   + self.a3 * Acceleration_old)
            F_hat = self.T.T @ F_hat

            #solve system for displacement
            Displacement_new[1:] = self.T[1:,1:] @ np.linalg.solve(K_hat[1:,1:], F_hat[1:])

            #update acceleration using Newmark coefficients
            Acceleration_new = self.a0 * (Displacement_new - Displacement_old)  - self.a2*Velocity_old -  self.a3 * Acceleration_old

            #update velocity using Newmark coefficients
            Velocity_new = Velocity_old + self.a6 * Acceleration_old + self.a7*Acceleration_new

            self.SnapshotsMatrixDisplacements[:,i] = Displacement_new
            self.SnapshotsMatrixVelocities[:,i] = Velocity_new
            self.SnapshotsMatrixAccelerations[:,i] = Acceleration_new

            #reset_variables
            Displacement_old = Displacement_new.copy()
            Velocity_old = Velocity_new.copy()
            Acceleration_old = Acceleration_new.copy()


    def set_up_Newmark_coefficients(self, alpha=0.25, beta=0.5):
        dt = self.time_step_size
        self.a0=1/(alpha*(dt**2))
        self.a1=beta/(alpha*dt)
        self.a2=1/(alpha*dt)
        self.a3=(1/(2*alpha))-1
        self.a4=(beta/alpha)-1
        self.a5=(dt/2)*((beta/alpha)-2)
        self.a6=dt*(1-beta)
        self.a7=beta*dt

#Launching the FOM simulation

number_of_elements = 500
total_time = 5
time_step_size = 0.002
element_degree = 1

fom_simulation = FOM_simulation(number_of_elements, total_time,time_step_size,element_degree) #creates the domain and the reference element

print('FOM simulation with ', fom_simulation.domain.number_of_elements, ' elements')
print('FOM simulation with ', fom_simulation.number_of_time_steps, ' time steps')

#Running the Simulation

tiF = time.time()
fom_simulation.Run()  # creates the system matrix K and solves the system for each of the load steps
tfF = time.time()
print('Time for the FOM simulation =', tfF - tiF)

class ROM_simulation(FOM_simulation):


    def __init__(self, number_of_elements,total_time,time_step_size,element_degree,basis):
        super().__init__(number_of_elements, total_time, time_step_size,element_degree)
        self.basis = basis

    def ComputeSystemMatrix(self):

        T = csr_matrix(fom_simulation.T)
        master_rows = self.find_master_rows(T)
        
        pTPhi = self.basis[master_rows,:]

        self.matrixD = fom_simulation.T @ pTPhi

        self.K1 = self.matrixD.T @ fom_simulation.K @ self.matrixD
        self.M1 = self.matrixD.T @ fom_simulation.M @ self.matrixD

    def find_master_rows(self, matrix):
        vect = [] 
        mask = np.zeros(matrix.shape[1]) 
        row = 0
        row_length = matrix.indptr[1]
        for idx, col in enumerate(matrix.indices):
            if idx == matrix.indptr[row+1]:
                row += 1
                row_length = matrix.indptr[row+1] - matrix.indptr[row]
            # print(f"row {row}, col {col}, val {matrix.data[idx]}")
            if not(mask[col]) and row_length == 1:
                vect.append(row)
                mask[col] = 1.0
        return vect


    def GetInitialDisplacement(self, applied_force):
        # This method solves a static problem to get the initial displacement at all nodes.
        applied_force_vector = np.zeros((self.domain.number_of_nodes))
        applied_force_vector[-1] = applied_force
        applied_force_vector_rom = self.matrixD.T @ applied_force_vector
        Displacement_old_rom = self.matrixD @ np.squeeze(np.linalg.solve(self.K1 , applied_force_vector_rom))
        return Displacement_old_rom


    def Solve(self):
        self.set_up_Newmark_coefficients()

        applied_force = 1
        Displacement_old_rom = self.GetInitialDisplacement(applied_force) #solve static problem to determine original deformation state

        Displacement_new_rom = Displacement_old_rom.copy()
        Velocity_old_rom = np.zeros((np.shape(self.basis)[0]))
        Acceleration_old_rom = np.zeros((np.shape(self.basis)[0]))

        self.SnapshotsMatrixDisplacements_rom = np.zeros((np.shape(self.basis)[0],self.number_of_time_steps))
        self.SnapshotsMatrixDisplacements_rom[:,0] = Displacement_old_rom
        self.SnapshotsMatrixVelocities_rom = np.zeros((np.shape(self.basis)[0],self.number_of_time_steps))
        self.SnapshotsMatrixVelocities_rom[:,0] = Velocity_old_rom
        self.SnapshotsMatrixAccelerations_rom = np.zeros((np.shape(self.basis)[0],self.number_of_time_steps))
        self.SnapshotsMatrixAccelerations_rom[:,0] = Acceleration_old_rom

        F = np.zeros((np.shape(fom_simulation.M)[1]))
        K_hat = self.K1 + self.a0*self.M1

        for i in range(1,self.number_of_time_steps):
            F_hat = F + fom_simulation.M @ (self.a0*Displacement_old_rom  +  self.a2*Velocity_old_rom  + self.a3 * Acceleration_old_rom)
            F_hat = self.matrixD.T @ F_hat

            #solve system for displacement
            Displacement_new_rom  = self.matrixD @ np.linalg.solve(K_hat, F_hat)

            #update acceleration using Newmark coefficients
            Acceleration_new_rom = self.a0 * (Displacement_new_rom - Displacement_old_rom)  - self.a2*Velocity_old_rom -  self.a3 * Acceleration_old_rom

            #update velocity using Newmark coefficients
            Velocity_new_rom = Velocity_old_rom + self.a6 * Acceleration_old_rom + self.a7*Acceleration_new_rom

            self.SnapshotsMatrixDisplacements_rom[:,i] = Displacement_new_rom
            self.SnapshotsMatrixVelocities_rom[:,i] = Velocity_new_rom
            self.SnapshotsMatrixAccelerations_rom[:,i] = Acceleration_new_rom

            #reset_variables
            Displacement_old_rom = Displacement_new_rom.copy()
            Velocity_old_rom = Velocity_new_rom.copy()
            Acceleration_old_rom = Acceleration_new_rom.copy()

# Proper Orthogonal Decomposition

from matplotlib.rcsetup import validate_aspect

def compute_basis(SnapshotsMatrix, truncation_tolerance=1e-4):

    #take the svd ignoring fixed dof
    u,s,v = np.linalg.svd(SnapshotsMatrix[1:,:],full_matrices=False)

    norm_S = np.linalg.norm(SnapshotsMatrix)

    m = np.size(s)
    n = m

    for k in range(1,m):
      numerator = 0
      denominator = 0
      for i in range(k,m):
        numerator += s[i]**2
      for j in range(0,k):
        denominator += s[j]**2
      if np.sqrt(numerator/denominator) <= (truncation_tolerance * norm_S):
        n = k
        break

    Phi = u[:,:n]

    return np.r_[np.zeros((1,Phi.shape[1])), Phi] #return adding ignored dofs

# call your function and obtain a basis from the FOM simulation snapshots

basis = compute_basis(fom_simulation.SnapshotsMatrixDisplacements, 5e-7)

print('\n\nThe basis shape is: ', basis.shape, '\n\n')

#create the ROM simulation with the same parameters as the FOM, plus a basis ( POD )

rom_simulation = ROM_simulation(number_of_elements,total_time, time_step_size, element_degree, basis)

tiR = time.time()
rom_simulation.Run()
tfR = time.time()
print('Time for the ROM simulation =', tfR - tiR)

FOM = fom_simulation.SnapshotsMatrixDisplacements
ROM = rom_simulation.SnapshotsMatrixDisplacements_rom

#compute the approximation error

print('approximation error: ', 100* np.linalg.norm(FOM - ROM)/ np.linalg.norm(FOM), ' %' )

# Attenzione che sono matrici rettangolari quindi il numero di condizionamento non potrebbe essere calcolato
# print('The condition number of the relation T^T Phi is', np.linalg.cond(fom_simulation.T.T @ rom_simulation.basis))
# print('The condition number of the relation (T^T T)^{-1} T^T Phi is ', np.linalg.cond(np.linalg.inv(fom_simulation.T.T @ fom_simulation.T) @ fom_simulation.T.T @ rom_simulation.basis))

def plot_a_snapshot(snapshots_FOM, snapshots_ROM, snapshot_to_print):
  snapshot_i_FOM = snapshots_FOM[:,snapshot_to_print]
  snapshot_i_ROM = snapshots_ROM[:,snapshot_to_print]
  number_of_nodes = np.shape(snapshot_i_FOM)[0]
  x_axis = np.linspace(0,number_of_nodes,number_of_nodes)
  plt.plot(x_axis, snapshot_i_FOM, 'b-o',alpha=0.5,label='FOM')
  plt.plot(x_axis, snapshot_i_ROM,'ro',label='ROM')
  plt.ylabel('displacement')
  plt.xlabel('node')
  plt.title(f'FOM vs ROM Simulation Results', fontsize=15, fontweight='bold')
  plt.legend()
  plt.show()

snapshot_to_print = 15
plot_a_snapshot(FOM,ROM, snapshot_to_print)
