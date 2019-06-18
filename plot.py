import matplotlib.pyplot as plt
from sklearn import externals
from sklearn.datasets.samples_generator import make_blobs
from sklearn import decomposition


model_filename = 'dlv-clusters.joblib.z'
vec_filename = 'dlv-vectorizer.joblib.z'
labels_filename = 'dlv-labels.joblib.z'
clusters_filename = 'dlv-clusteritems.joblib.z'
model = externals.joblib.load(model_filename)
vectorizer = externals.joblib.load(vec_filename)
labels = externals.joblib.load(labels_filename)
clusters = externals.joblib.load(clusters_filename)


X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()
