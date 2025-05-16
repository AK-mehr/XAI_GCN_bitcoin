import tensorflow as tf
import numpy as np
import scipy.sparse as sp


def convert_scipy_CRS_space_to_tensor(sparce_mat: sp.csr.csr_matrix):
    sparce_mat_coo = sparce_mat.tocoo()
    indices = np.transpose(np.array([sparce_mat_coo.row, sparce_mat_coo.col]))
    return tf.SparseTensor(indices, sparce_mat_coo.data, sparce_mat_coo.shape)


# def normalize_adjencency_mat(adj_mat: sp.csr.csr_matrix):
#     assert len(adj_mat.shape) == 2
#     a = a.tocoo()  # if not already sparse
#     d = np.array(adj_mat.sum(axis=-1))[...,0]
#     d = np.sqrt(d)
#     d = np.array(d).flatten()  # ← ensure it's an array!
#     d = sp.diags(d).tocsr()
#     a = adj_mat + sp.eye(adj_mat.shape[-1],dtype=adj_mat.dtype)
#     return d @ a @ d
import scipy.sparse as sp
import numpy as np

def normalize_adjencency_mat(adj_mat: sp.csr.csr_matrix):
    assert len(adj_mat.shape) == 2

    # Add self-loops to the adjacency matrix
    a = adj_mat + sp.eye(adj_mat.shape[0], dtype=adj_mat.dtype)

    # Compute node degrees
    d = np.array(a.sum(axis=1)).flatten()

    # Compute D^(-1/2)
    d = np.power(d, -0.5)
    d[np.isinf(d)] = 0.  # Avoid division by zero
    d = sp.diags(d)

    # Return normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
    return d @ a @ d
