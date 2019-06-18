from sklearn import externals
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words

model_filename = 'dlv-clusters.joblib.z'
vec_filename = 'dlv-vectorizer.joblib.z'
labels_filename = 'dlv-labels.joblib.z'
clusters_filename = 'dlv-clusteritems.joblib.z'
model = externals.joblib.load(model_filename)
vectorizer = externals.joblib.load(vec_filename)
labels = externals.joblib.load(labels_filename)
clusters = externals.joblib.load(clusters_filename)


def predict(text):
    Y = vectorizer.transform([text])
    prediction = model.predict(Y)
    # print(labels)
    # print(prediction)
    # print(clusters)
    # print("\n")
    # print "Cluster ", prediction
    # print(clusters[prediction[0]][:10])
    return {'Cluster': prediction[0].astype(int),
            'DLV 1': clusters[prediction[0]][0],
            'DLV 2': clusters[prediction[0]][1],
            'DLV 3': clusters[prediction[0]][2],
            'DLV 4': clusters[prediction[0]][3],
            'DLV 5': clusters[prediction[0]][4]}
