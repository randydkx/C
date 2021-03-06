from sklearn.datasets.samples_generator import make_blobs
import sys
sys.path[0] = '../'
from ml_models import utils

X, y = make_blobs(n_samples=400, centers=4, cluster_std=0.85, random_state=0)
X = X[:, ::-1]

from ml_models.pgm import GaussianNBClassifier

nb = GaussianNBClassifier()
nb.fit(X, y)
print(nb.predict(X))
utils.plot_decision_function(X, y, nb)
utils.plt.show()
