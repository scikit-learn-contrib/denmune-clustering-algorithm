import numpy as np
import glob # for using chain 
import pytest
from sklearn.datasets import make_blobs
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
def test_parameters(train_data, train_truth, test_data, test_truth, validate, show_plots, show_noise, show_analyzer):
    if not (train_data is None):
        if not (train_data is not None and train_truth is None and test_truth is not None):
            if not (train_data is not None and test_data is not None and train_truth is None):
                 if not (train_data is not None and  train_truth is not None and test_truth is not None  and test_data is None):
                    dm = DenMune(train_data=train_data, train_truth=train_truth, test_data=test_data, test_truth=test_truth, k_nearest=10)
                    labels, validity = dm.fit_predict(validate=validate, show_plots=show_plots, show_noise=show_noise, show_analyzer=show_analyzer)
                    # This test use data that are not perfectly separable so the
                    # accuracy is not 1. Accuracy around 0.70
                    assert ( np.mean(dm.labels_pred == y_cc) > 0.80 or (1 - np.mean( dm.labels_pred == y_cc)  > 0.80) ) 


def test_DenMune_propagation():
    snapshots = chain([0], range(2,5), range(5,50,5), range(50, 100, 10), range(100,500,50), range(500,1100, 100))
    for snapshot in snapshots:
        dm = DenMune(train_data=X_cc, k_nearest=knn, prop_step=snapshot)
        labels, validity = dm.fit_predict(show_analyzer=False, show_plots=False)   
    assert (snapshot == 1000)
