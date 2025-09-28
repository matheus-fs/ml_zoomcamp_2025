'''
In this notebook I implement some basic linear algebra functions and compare the results with numpy
- Vector x Vector multiplication
- Vector x Matrix multiplication
- Matrix x Matrix multiplication
'''


#%%
from typing import List
import numpy as np

Vector = List[float]
Matrix = List[Vector]

# Vector operations - dot product
def vector_multiply(u: Vector, v: Vector) -> float:

    assert len(u) == len(v)

    result = 0.0
    for i in range(len(u)):
        result += u[i] * v[i]
    
    return result

# testing
u = [1.0, 2.0, 3.0]
v = [1.0, 2.1, 3.0]

np_u = np.array(u)
np_v = np.array(v)

assert vector_multiply(u, v) == 14.2
assert vector_multiply(u,v) == np_u.dot(v)


#%%
def vector_matrix_multiply(U: Matrix, v: Vector) -> Vector:

    assert len(U[0]) == len(v)

    result = [vector_multiply(U[i], v) for i in  range(len(U))]

    return result

U = [
    [1.0, 1.0, 5.0],
    [5.0, 4.0, 6.0],
    [7.0, 6.0, 8.0],
    [3.0, 5.0, 7.0]
]
v = [1.0, 2.0, 3.0]

np_U = np.array(U)
np_v = np.array(v)

assert vector_matrix_multiply(U, v) == np_U.dot(np_v).tolist()


#%%
def matrix_multiply(U: Matrix, V: Matrix) -> Matrix:
    '''
    Number of columns in the first matrix must be the same as number of rows os the second matrix
    Result matrix must have number of rows from the first matrix and number of columns of second matrix
    '''

    assert len(U[0]) == len(V)

    # Initializing the result matrix
    rows = [0.0 for _ in range(len(V[0]))]
    result = [rows.copy() for _ in range(len(U))]

    for row in range(len(U)): #Loop over rows
        for col in range(len(V[0])): # Loop over columns

            result[row][col] = vector_multiply(U[row], [i[col] for i in V])

    return result

U = [
    [2., 4., 5., 6.],
    [1., 2., 1., 2.],
    [3., 1., 2., 1.]
]

V = [
    [1., 1., 2.],
    [0., 0.5, 1.],
    [0., 2., 1.],
    [2., 1., 0.],
]

np_U = np.array(U)
np_V = np.array(V)

assert matrix_multiply(U, V) == np_U.dot(np_V).tolist()

# %%
