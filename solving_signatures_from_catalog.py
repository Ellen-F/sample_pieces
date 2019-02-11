import warnings
warnings.filterwarnings("ignore")

import pickle
import scipy
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from sklearn.decomposition import non_negative_factorization
from sklearn.decomposition import ProjectedGradientNMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib
matplotlib.use('Agg')  # hpc
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # hpc
import matplotlib.cm as cm
import numpy as np


# loading
all_data = sio.loadmat('for_input_Tpaired_finalFA_192_noextreme_re.mat')
V = all_data['originalGenomes']
print(all_data.keys())
print('target matrix shape is:{0}'.format(V.shape))


# tweak
V = V+10e-8


# constants
nb_workers=56
nb_iter_per_core = 5
nb_max_sig = 30
print('[para]iter per core:{0}'.format(nb_iter_per_core))
print('[para]work on {0} cores'.format(nb_workers))


# worker funcs
def worker(V, args, verbose=1):
    W_all = []
    H_all = []
    recons_error = []
    for i in range(args['nb_iter_per_core']):
        boot_V = bootstrapGenomes(V)
        W, H = decomposition(boot_V, args['W'], args['H'], args['n_components'],
                             solver=args['solver'], update_H=args['update_H'])
        W_all.append(W)
        H_all.append(H)
        recons_error.append(boot_V - np.dot(W, H))
        if verbose:
            print('[__worker__]:Iter {0}/{1} to extract {2} sigs completed.'.format(i, args['nb_iter_per_core'],
                                                                                args['n_components']))
    W_all = np.asarray(W_all)
    H_all = np.asarray(H_all)

    recons_error = np.asarray(recons_error)
    return W_all, H_all, recons_error
        

def decomposition(V, W, H, n_components,
                  solver='mu', update_H=True):
    if solver!='project':
        W, H, _ = non_negative_factorization(V, W=W, H=H, n_components=n_components,
                                             update_H=update_H, max_iter=1000, solver=solver)
                                             #regularization='transformation', l1_ratio=0.1)
    else:
        model = ProjectedGradientNMF(n_components=n_components, init='random', 
                                     random_state=0, sparseness='data', beta=0,
                                     max_iter=100000)
        model.fit(V)
        H = model.components_
        W = model.fit_transform(V)
    return W, H


def normalize_genomes(X):
    repmat = np.matlib.repmat(np.sum(X, axis=0),
                              X.shape[0],
                              1)
    X = X / repmat
    return X


def remove_weak_rows(X):
    sorted_X = np.sort(X, axis=0)
    sorted_X_index = np.argsort(X, axis=0)
    total_mutation_count = np.sum(np.sum(X))
    total_mutation_toremove = np.sum(np.cumsum(sorted_X, axis=0)/total_mutation_count < 0.01)
    toremove_set = sorted_X_index[1:total_mutation_toremove]
    
    X[toremove_set, :] = 10e-7
    return X


def bootstrapGenomes(X, n=None):
    normX = normalize_genomes(X)
    normX = remove_weak_rows(normX)
    if n == None:
        n = len(X)
    #resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    #X_resample = X[resample_i]
    bootstrap_all = []
    for i in range(X.shape[1]):
        N = np.rint(np.sum(X, axis=0))
        bootstrapX = np.random.multinomial(N[i], normX[:, i])
        bootstrap_all.append(bootstrapX)
    bootstrap_all = np.asarray(bootstrap_all).T
    return bootstrap_all


def evaluateStability(W_all, H_all, distance_measure='cosine', method='silhouette'):
    """
    parameters
        W_all:: matrix, shape = nb_workers*nb_iter_per_core x 96 x n_components
        H_all:: matrix, shape = nb_workers*nb_iter_per_core x n_components x nb_samples
    """
    if method=='mine':
        # my arbitrary method
        similarity_list = []
        for i in range(W_all.shape[0]-1):
            for j in range(i+1, W_all.shape[0]):
                if distance_measure=='cosine':
                    similarity = cosine_similarity(W_all[i], W_all[j])
                similarity = np.linalg.norm(similarity)
                similarity_list.append(similarity)
        similarity_list = np.asarray(similarity_list)
        print('mean:', np.mean(similarity_list))
        print('std:', np.std(similarity_list))
        return similarity_list
    elif method=='silhouette':
        nb_totaliters, nb_features, n_components = W_all.shape
        nb_samples = H_all.shape[-1]
        W_all = np.squeeze(W_all.reshape(nb_features, n_components*nb_totaliters))
        H_all = H_all.reshape(n_components*nb_totaliters, nb_samples)
        print(W_all.shape)
        print(H_all.shape)
        
        which_all = H_all
        n_cluster = n_components
        
        clusterer = KMeans(n_clusters=n_cluster, random_state=10)
        cluster_labels = clusterer.fit_predict(which_all)
        cluster_centers = clusterer.cluster_centers_
        silhouette_avg = silhouette_score(which_all, cluster_labels)
        sample_silhouette_values = silhouette_samples(which_all, cluster_labels)
        return silhouette_avg, sample_silhouette_values, cluster_labels, which_all, clusterer, cluster_centers
    

def plot_silhouette(X, sample_silhouette_values, silhouette_avg,
                    cluster_labels, n_cluster, clusterer):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    y_lower = 10
    for i in range(n_cluster):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_cluster)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = cm.spectral(cluster_labels.astype(float) / n_cluster)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_cluster),
                 fontsize=14, fontweight='bold')
    plt.savefig('TCGAlegacy_finalFA192_TUMOR_restricted_{0}_re.png'.format(n_cluster), dpi=600)
    
    
def evaluate_model(V, p, nb_max_sig, nb_iter_per_core, nb_workers):
    """
    p:: Pool(nb_workers) object
    nb_max_sig:: integer
    nb_iter_per_core:: as the name indicates
    nb_workers:: as the name indicates
    """
    for n_components in range(19,nb_max_sig):
        print('[loop]extracting {0} signatures'.format(n_components))
        parameters = {'W':None, 'H':None, 'n_components':n_components,
                      'solver':'project', 'update_H':True,
                      'nb_iter_per_core':nb_iter_per_core}

        output_by_workers = p.starmap(worker, zip([V]*nb_workers, 
                                                  [parameters]*nb_workers,
                                                  [0]*nb_workers))

        W_all = np.asarray([_[0] for _ in output_by_workers]).reshape(nb_workers*nb_iter_per_core,
                                                                      192, n_components)
        H_all = np.asarray([_[1] for _ in output_by_workers]).reshape(nb_workers*nb_iter_per_core,
                                                                      n_components, output_by_workers[0][1].shape[-1])
        recons_error = [_[2] for _ in output_by_workers]
        silhouette_avg, sample_silhouette_values, cluster_labels, which_all, clusterer, centers = evaluateStability(W_all, H_all, 
                                                                                                            method='silhouette')
        plot_silhouette(which_all, sample_silhouette_values, silhouette_avg,
                        cluster_labels, n_cluster=n_components, clusterer=clusterer)

        # save files
        W = np.sum(W_all, axis=0)/W_all.shape[0]
        H = np.sum(H_all, axis=0)/H_all.shape[0]
        recons_error = np.sum(np.asarray(recons_error), axis=0)
        pickle.dump([W, H, recons_error, silhouette_avg], 
                    open('py_TCGAlegacy_finalFA192_TUMORsig{0}_re.pkl'.format(n_components),'wb')
                   )
        
    return W, H, recons_error
    
    
if __name__=='__main__':
    warnings.filterwarnings("ignore")
    with Pool(nb_workers) as p:
        evaluate_model(V, p, nb_max_sig, nb_iter_per_core, nb_workers)

            




