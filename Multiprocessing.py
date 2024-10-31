"""
README:
This code performs matrix-vector multiplication using parallel processes. 
The Pool object from the multiprocessing module is used to distribute rows of the matrix among
worker processors and multiply the assigned row by the vector.
"""


from multiprocessing import Pool
import numpy as np

#Data paralelisim using pool
n_p=5 # Number of Procesors
n = 100 # my matrix and vector will choose random integers between [0,n)
n_rows = 13  # Number of rows in the matrix
n_cols = 12  # Number of columns in the matrix = Vector Dimension
vector = np.random.randint(0, n, size=n_cols) # Defining my random vector
matrix = np.random.randint(low=0, high=n, size=(n_rows, n_cols)) # Defining my matrix

def f(x):
    return(np.dot(x,vector))

if __name__ == '__main__':
    with Pool(n_p) as p: #Pool of 5 worker processes
        print(np.array(p.map(f, matrix))) # map distripute the task among processes
