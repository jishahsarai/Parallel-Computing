"""
README
This code performs matrix-vector multiplication using parallel processes. 
In the INPUT section, you can configure the following parameters:
    - Seed: Set the random number generator seed.
    - n: Define the upper limit for random number generation.
    - Dimensions: Specify the number of rows (n_rows) and columns (n_cols), used in the matrix and vector.
This code is designed to run with Windows MPI. To execute the code, use the following command:
mpiexec -n 4 python Problem2.py

"""

from mpi4py import MPI
import numpy as np

##MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#INPUT
np.random.seed(42) # Seed
n=100 # my matrix and vector will choose random integers between [0,n)
n_rows = 10  # Number of rows in the matrix
n_cols = 8  # Number of columns in the matrix = Vector Dimension

#Initialize matrix and vector
if rank == 0: #ROOT
    vector = np.random.randint(0, n, size=n_cols) # Defining my random vector
    matrix = np.random.randint(low=0, high=n, size=(n_rows, n_cols)) # Defining my matrix
    piece = np.array_split(matrix, size, axis=0) # np.array_split - SPLITS the matrix as evenly as possible among the processors
else:
    vector = None
    matrix = None
    piece = None

# BROADCAST the vector
vector = comm.bcast(vector, root=0)

# SCATTER the matrix among processes
local_matrix = comm.scatter(piece, root=0)

# LOCAL matrix*vector multiplication
partial_result = np.dot(local_matrix, vector)

# GATHER RESULTS
result = comm.gather(partial_result, root=0)
if rank == 0:
    result = np.concatenate(result) #Concatenate the results because each process returns an array with 1 element
    print("Matrix:\n", matrix) 
    print("Vector:\n", vector)
    print("Result of Matrix-Vector Multiplication:\n", result) #Print my result


