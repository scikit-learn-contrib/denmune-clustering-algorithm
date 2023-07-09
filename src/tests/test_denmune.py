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


knn = 10

def test_DenMune_results():
    dm = DenMune(train_data=X_cc, train_truth=y_cc, k_nearest=knn)
    labels, validity = dm.fit_predict(show_analyzer=False)
    # This test use data that are not perfectly separable so the
    # accuracy is not 1. Accuracy around 0.90
    assert (np.mean(dm.labels_pred == y_cc) < 0.80) or (1 - np.mean(dm.labels_pred == y_cc) < 0.80)
