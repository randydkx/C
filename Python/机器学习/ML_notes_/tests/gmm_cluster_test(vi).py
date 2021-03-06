from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.55, random_state=0)
X = X[:, ::-1]

from ml_models.vi import GMMCluster

gmm = GMMCluster(verbose=True, n_iter=100, n_components=5)
gmm.fit(X)
