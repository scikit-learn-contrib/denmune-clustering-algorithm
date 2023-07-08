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


@pytest.mark.parametrize("train_data", [None, train_data ])
@pytest.mark.parametrize("train_truth", [None, train_labels ])  
@pytest.mark.parametrize("test_data", [None, test_data ])  
@pytest.mark.parametrize("test_truth", [None, test_labels ])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("show_plots", [True, False])
@pytest.mark.parametrize("show_noise", [True, False])
@pytest.mark.parametrize("show_analyzer", [True, False])
@pytest.mark.parametrize("prop_step", [0, 0])

# all possible combinations will be tested over all parameters. Actually, 257 tests will be covered
def test_parameters(train_data, train_truth, test_data, test_truth, validate, show_plots, show_noise, show_analyzer, prop_step):
    if not (train_data is None):
        if not (train_data is not None and train_truth is None and test_truth is not None):
            if not (train_data is not None and test_data is not None and train_truth is None):
                 if not (train_data is not None and  train_truth is not None and test_truth is not None  and test_data is None):
                    dm = DenMune(train_data=train_data, train_truth=train_truth, test_data=test_data, test_truth=test_truth, k_nearest=10,prop_step=prop_step)
                    labels, validity = dm.fit_predict(validate=validate, show_plots=show_plots, show_noise=show_noise, show_analyzer=show_analyzer)
                    # This test use data that are not perfectly separable so the
                    # accuracy is not 1. Accuracy around 0.70
                    assert ( np.mean(dm.labels_pred == y_cc) > 0.70 or (1 - np.mean( dm.labels_pred == y_cc)  > 0.70) ) 
