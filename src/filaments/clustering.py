import numpy as np

def prepare_clustering(chl, temp):
    return np.array(list(zip(chl[~np.isnan(chl)], temp[~np.isnan(chl)])))

def labels_matrix(labels, mask):
    labels_matrix = np.empty(mask.shape)
    labels_matrix[:,:] = np.nan
    labels_matrix[mask] = labels.astype(int)
    return labels_matrix
