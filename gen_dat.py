import numpy as np
import scipy.stats as stats
from scipy.stats import ortho_group

def gen_data(n, p, K, eigen=None, verbose=True):
    """
    Generates data with clustered individualized treatment effects (ITEs)
    
    Parameters
    -------
    n : int
        Number of points
        
    p : int
        Number of features
        
    K : int
        Number of clusters
        
    eigen : array (optional)
        p x 1 array of eigenvalues for the covariance matrix of the Gaussian distribution
    
    verbose : bool (optional)
        If True, prints the number of points in each cluster
        
    Returns
    -------
    X : array
        n x p array of features
    
    y : array
        n x 1 array of individualized treatment effects (bounded between -1 and +1)
        
    cluster_real_cat : array
        n x 1 array of cluster labels
    
    cluster_prob : array
        n x K array of probabilities of belonging to each of the K clusters
    """
    
    # Sample a pxp orthonormal matrix
    orth_mat = ortho_group.rvs(dim=p)

    # Create a positive semi-definite matrix with p eigenvalues all between 0 and 1
    sigma_mat = orth_mat @ np.diag(np.random.uniform(0, 1, p)) @ orth_mat.T

    # Sample points from a p-dimentional Gaussian
    if eigen is not None:
        X = np.random.multivariate_normal(mean=eigen, cov=sigma_mat, size=n)
    else:    
        X = np.random.multivariate_normal(mean=[0] * p, cov=sigma_mat, size=n)

    # Create parameters for the probability of belonging to each of the K clusters
    cl_param = np.random.uniform(0, 1, (K,p))

    # Create these probabilities for each point
    def softmax(X, cl_param, k):
        return np.exp(X @ cl_param[k]) / np.sum([np.exp(X @ cl_param[i]) for i in range(cl_param.shape[0])], axis=0)

    def all_softmax(X, cl_param):
        return np.array([softmax(X, cl_param, k) for k in range(cl_param.shape[0])]).T

    cluster_prob = all_softmax(X, cl_param)

    # For the real clusters, sample one hot encoded random vectors from a multinomial distribution
    cluster_real_oh = np.array([np.random.multinomial(n=1, pvals=cluster_prob[i]) for i in range(cluster_prob.shape[0])])

    # Convert one hot encoding to categorical
    cluster_real_cat = cluster_real_oh.argmax(axis=1)

    # Create parameters for each of the K experts
    ex_param = np.random.uniform(0, 1, (K,p))

    # Create all predictions for the K experts for all n points
    # To mimic individualized treatment effects (ITEs), bound the predictions between -1 and +1 like so
    ex_preds = 2 * np.array([1 / (1 + np.exp(-X @  ex_param[ex])) for ex in range(ex_param.shape[0])]) - 1 

    # Pick as real ITE the prediction from the corresponding cluster
    # by applying the one-hot encoding mask like so
    y = np.sum(ex_preds.T * cluster_real_oh, axis=1)

    if verbose:
        # Number of points in each cluster
        for it, count in enumerate(np.unique(cluster_real_cat, return_counts=True)[1]):
            print(f'cluster {it+1}: {count} points')        
    
    return X, y, cluster_real_cat, cluster_prob