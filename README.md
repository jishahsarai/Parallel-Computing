# Matrix-Vector Multiplication with Parallel Processing
This repository includes two Python programs that perform matrix-vector multiplication using parallel processing techniques. Each program utilizes a different parallel processing method to speed up computations by distributing tasks among multiple processors.

## Requirements
- Python 3.x
- NumPy for matrix and vector operations
- mpi4py library for MPI communication
  Microsoft MPI or any compatible MPI implementation
- Multiprocessing library

## Code Overview
1. Matrix-Vector Multiplication with Multiprocessing
This code uses Python's multiprocessing library to perform parallel matrix-vector multiplication on a single machine. The main components are:
- Multiprocessing Pool: Creates a pool of worker processes that share tasks.

2. Matrix-Vector Multiplication with MPI
This code uses the mpi4py library, which provides MPI (Message Passing Interface) support for Python. This program is designed to be distributed across multiple processors, potentially on different machines, making it suitable for high-performance computing clusters.

**Key Components:**
- MPI Communication: Uses MPI.COMM_WORLD for communication between processes.
- Broadcast: Sends the vector from the root process to all other processes.
- Scatter: Divides the matrix across processes.
- Gather: Collects each process’s partial result and concatenates it to form the final result.
