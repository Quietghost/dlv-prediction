from flask import Flask
from flask import jsonify
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
app = Flask(__name__)


@app.route('/predict/<text>', methods=['POST'])
def predict(text):
    Y = vectorizer.transform([text])
    prediction = model.predict(Y)

    return jsonify(
        Cluster=prediction[0].astype(int),
        DLV_1=clusters[prediction[0]][0],
        DLV_2=clusters[prediction[0]][1],
        DLV_3=clusters[prediction[0]][2],
        DLV_4=clusters[prediction[0]][3],
        DLV_5=clusters[prediction[0]][4]
    )
