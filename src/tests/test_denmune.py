import numpy as np
from itertools import chain
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from src.denmune import DenMune

       
# test DenMune's results
X_cc, y_cc = make_blobs(
    n_samples=1000,
    centers=np.array([[-1, -1], [1, 1]]),
    random_state=0,
    shuffle=False,
    cluster_std=0.5,
)

data_file = 'https://raw.githubusercontent.com/egy1st/datasets/dd90854f92cb5ef73b4146606c1c158c32e69b94/denmune/shapes/aggr_rand.csv'
data = pd.read_csv(data_file, sep=',', header=None)
labels = data.iloc[:, -1]
data = data.drop(data.columns[-1], axis=1)
train_data = data [:555]
test_data = data [555:]
train_labels = labels [:555]
test_labels = labels [555:]



knn = 10

def test_DenMune_results():
    dm = DenMune(train_data=X_cc, train_truth=y_cc, k_nearest=knn)
    labels, validity = dm.fit_predict(show_analyzer=False)
    # This test use data that are not perfectly separable so the
    # accuracy is not 1. Accuracy around 0.90
    assert (np.mean(dm.labels_pred == y_cc) > 0.80) or (1 - np.mean(dm.labels_pred == y_cc) > 0.80)


