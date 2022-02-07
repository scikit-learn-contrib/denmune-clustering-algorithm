import numpy as np
import pytest
from sklearn.datasets import make_blobs
from denmune import DenMune

# test DenMune's results
X_cc, y_cc = make_blobs(
    n_samples=1000,
    centers=np.array([[-1, -1], [1, 1]]),
    random_state=0,
    shuffle=False,
    cluster_std=0.5,
)



def test_DenMune_results():
    dm = DenMune(train_data=X_cc, train_truth=y_cc, k_nearest=10)
    labels, validity = dm.fit_predict(show_analyzer=False)
    # This test use data that are not perfectly separable so the
    # accuracy is not 1. Accuracy around 0.90
    assert (np.mean(dm.labels_pred == y_cc) > 0.90) or (1 - np.mean(dm.labels_pred == y_cc) > 0.90)
    
@pytest.mark.parametrize("validate", [True, False])  
@pytest.mark.parametrize("show_plots", [True, False])  
@pytest.mark.parametrize("show_noise", [True, False])  
@pytest.mark.parametrize("show_analyzer", [True, False])  
def test_fit_predict_parameters(validate, show_plots, show_noise, show_analyzer):
    dm = DenMune(train_data=X_cc, train_truth=y_cc, k_nearest=10)
    labels, validity = dm.fit_predict(validate=validate, show_plots=show_plots, show_noise=show_noise, show_analyzer=show_analyzer)
    # This test use data that are not perfectly separable so the
    # accuracy is not 1. Accuracy around 0.90
    assert (np.mean(dm.labels_pred == y_cc) > 0.90) or (1 - np.mean(dm.labels_pred == y_cc) > 0.90)    

    
@pytest.mark.parametrize("train_data", [X_cc[:800] ])  
@pytest.mark.parametrize("train_truth", [None, y_cc[:800] ])  
@pytest.mark.parametrize("test_data", [None, X_cc[800:] ])  
@pytest.mark.parametrize("test_truth", [None, y_cc[800:] ])  
def test_init_parameters(train_data, train_truth, test_data, test_truth):
    dm = DenMune(train_data=train_data, train_truth=train_truth, test_data=test_data, test_truth=test_truth, k_nearest=10)
    labels, validity = dm.fit_predict()
    # This test use data that are not perfectly separable so the
    # accuracy is not 1. Accuracy around 0.70
    #assert (np.mean(dm.labels_pred == y_cc) > 0.70) or (1 - np.mean(dm.labels_pred == y_cc) > 0.70)    


