import torch
from utils import *



# def get_mol_dict(dg):
#     mol_dict = dict()
#     for v in dg.vertices:
#         mol_dict[v.graph.name] = v.graph.id
#     return mol_dict



# Function for tropical matrix multiplication
def tropical_multiply(mat1, mat2):
    """
    Performs tropical matrix multiplication.

    Parameters:
    - mat1 (torch.Tensor): First input matrix.
    - mat2 (torch.Tensor): Second input matrix.

    Returns:
    torch.Tensor: Result of the tropical matrix multiplication.
    """
    # Ensure the input matrices have compatible shapes
    assert mat1.shape[1] == mat2.shape[0], "Incompatible matrix dimensions for multiplication"

    result_rows, result_cols = mat1.shape[0], mat2.shape[1]
    result = torch.full((result_rows, result_cols), float('inf'))

    for i in range(result_rows):
        for j in range(result_cols):
            for k in range(mat1.shape[1]):
                result[i, j] = min(result[i, j], mat1[i, k] + mat2[k, j])

    return result


# Function for tropical matrix addition
def tropical_add(mat1, mat2):
    """
    Performs tropical matrix addition.

    Parameters:
    - mat1 (torch.Tensor): First input matrix.
    - mat2 (torch.Tensor): Second input matrix.

    Returns:
    torch.Tensor: Result of the tropical matrix addition.
    """
    # Ensure the input matrices have compatible shapes
    assert mat1.shape == mat2.shape, "Incompatible matrix dimensions for addition"

    result_rows, result_cols = mat1.shape
    result = torch.full((result_rows, result_cols), float('inf'))

    for i in range(result_rows):
        for j in range(result_cols):
            result[i, j] = min(mat1[i, j], mat2[i, j])

    return result


# Function to create a tropical identity matrix(tim) of given size
def tropical_identity(size):
    """
    Generates a tropical identity matrix of a given size.

    Parameters:
    - size (int): Size of the square identity matrix.

    Returns:
    torch.Tensor: Tropical identity matrix.
    """
    tim = torch.full((size, size), float('inf'))
    tim.fill_diagonal_(0.0)
    return tim


# Function implementing Torgan-Sin-Zimmerman algorithm
def torgansin_zimmerman(Q1, Q2):
    """
    Implements the Torgan-Sin-Zimmerman algorithm for tropical semiring operations.

    Parameters:
    - Q1 (torch.Tensor): Hyperedge(Rules) connect to Vertices(Products) incidence matrix.
    - Q2 (torch.Tensor): Verticies(Reactants) connect to Hyperedge(Rules) incidence matrix.

    Returns:
    Tuple of torch.Tensor:
    - P1_2m1: Result of all pairs shortest distance of left up block.
    - P2_2m1: Result of all pairs shortest distance of right bottom block.
    - Q1_2m1: Result of all pairs shortest distance of left bottom block.
    - Q2_2m1: Result of all pairs shortest distance of right up  block.
    - D_: Concatenated result matrix.
    """
    m1 = Q1.shape[0]
    I1 = tropical_identity(m1)
    I2 = tropical_identity(Q2.shape[0])

    D = I1
    D1 = tropical_add(tropical_multiply(Q1, Q2), I1)

    for _ in range(1, m1):
        D = tropical_multiply(D, D1)

    P1_2m1 = tropical_multiply(D, D1)
    P2_2m1 = tropical_add(tropical_multiply(tropical_multiply(Q2, D), Q1), I2)
    Q1_2m1 = tropical_multiply(D, Q1)
    Q2_2m1 = tropical_multiply(Q2, P1_2m1)

    D_ = torch.cat([torch.cat([P1_2m1, Q1_2m1], dim=1), torch.cat([Q2_2m1, P2_2m1], dim=1)], dim=0)

    return P1_2m1, P2_2m1, Q1_2m1, Q2_2m1, D_
