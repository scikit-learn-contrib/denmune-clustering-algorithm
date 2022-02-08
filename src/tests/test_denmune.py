import numpy as np
from itertools import chain
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
    assert (np.mean(dm.labels_pred == y_cc) > 0.90) or (1 - np.mean(dm.labels_pred == y_cc) > 0.90)

@pytest.mark.parametrize("train_data", [None, X_cc[:800] ])
@pytest.mark.parametrize("train_truth", [None, y_cc[:800] ])  
@pytest.mark.parametrize("test_data", [None, X_cc[800:] ])  
@pytest.mark.parametrize("test_truth", [None, y_cc[800:] ])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("show_plots", [True, False])
@pytest.mark.parametrize("show_noise", [True, False])
@pytest.mark.parametrize("show_analyzer", [True, False])
@pytest.mark.parametrize("prop_step", [0, 50])
# all possible combination will be tested over all parameters. Actually, 257 tests will be covered
def test_parameters(train_data, train_truth, test_data, test_truth, validate, prop_step, show_plots, show_noise, show_analyzer):
    if not (train_data is None):
        if not (train_data is not None and train_truth is None and test_truth is not None):
            if not (train_data is not None and test_data is not None and train_truth is None):
                 if not (train_data is not None and  train_truth is not None and test_truth is not None  and test_data is None):
                    dm = DenMune(train_data=train_data, train_truth=train_truth, test_data=test_data, test_truth=test_truth, k_nearest=10, prop_step=prop_step)
                    labels, validity = dm.fit_predict(validate=validate, show_plots=show_plots, show_noise=show_noise, show_analyzer=show_analyzer)
                    # This test use data that are not perfectly separable so the
                    # accuracy is not 1. Accuracy around 0.70
                    assert ( np.mean(dm.labels_pred == y_cc) > 0.80 or (1 - np.mean( dm.labels_pred == y_cc)  > 0.80) ) 


def test_DenMune_propagation():
    snapshots = chain([0], range(2,5), range(5,50,5), range(50, 100, 10), range(100,500,50), range(500,1100, 100))
    for snapshot in snapshots:
        dm = DenMune(train_data=X_cc, k_nearest=knn, prop_step=snapshot)
        labels, validity = dm.fit_predict(show_analyzer=False, show_plots=False) 
    # if snapshot iteration = 1000, this means we could propagate to the end properly    
    assert (snapshot == 1000)

# we are going to do some tests using iris data    
X_iris = load_iris()["data"]
y_iris = load_iris()["target"]

# we test t_SNE reduction by applying it on Iris dataset which has 4 dimentions.
@pytest.mark.parametrize("file_2d", [None, 'iris_2d.csv'])
@pytest.mark.parametrize("rgn_tsne", [True, False])


def test_t_SNE(rgn_tsne, file_2d):
    dm = DenMune(train_data=X_iris, train_truth=y_iris, k_nearest=knn, rgn_tsne=rgn_tsne, file_2d=file_2d)
    labels, validity = dm.fit_predict(show_analyzer=False, show_plots=False)
    assert (dm.data.shape[1] == 2) # this means it was reduced properly to 2-d using t-SNE

def test_knn():
    for k in range (5, 55, 5):
        dm = DenMune(train_data=X_iris, train_truth=y_iris, k_nearest=k, rgn_tsne=False)
        labels, validity = dm.fit_predict(show_analyzer=False, show_plots=False)
    #assert (k == 50) # this means we tested the algorithm works fine with several knn inputs    
    
