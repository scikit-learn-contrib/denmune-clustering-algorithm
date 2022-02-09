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
    assert (np.mean(dm.labels_pred == y_cc) > 0.90) or (1 - np.mean(dm.labels_pred == y_cc) > 0.90)


@pytest.mark.parametrize("train_data", [None, X_cc[:800] ])
@pytest.mark.parametrize("train_truth", [None, y_cc[:800] ])  
@pytest.mark.parametrize("test_data", [None, X_cc[800:] ])  
@pytest.mark.parametrize("test_truth", [None, y_cc[800:] ])
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("show_plots", [True, False])
@pytest.mark.parametrize("show_noise", [True, False])
@pytest.mark.parametrize("show_analyzer", [True, False])
@pytest.mark.parametrize("prop_step", [0, 600]) 

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

def test_exceptions():

    with pytest.raises(Exception) as execinfo:
        raise Exception('train data is None')
        dm = DenMune(train_data=None, k_nearest=10)
        labels, validity = dm.fit_predict()

    with pytest.raises(Exception) as execinfo:
        raise Exception('train_data is not None and train_truth is None and test_truth is not None')
        dm = DenMune(train_data=train_data, test_truth=test_truth, k_nearest=10)
        labels, validity = dm.fit_predict()  

    with pytest.raises(Exception) as execinfo:
        raise Exception('train_data is not None and test_data is not None and train_truth is None')
        dm = DenMune(train_data=train_data, test_data=test_data, k_nearest=10)
        labels, validity = dm.fit_predict()  

    with pytest.raises(Exception) as execinfo:
        raise Exception('train_data is not None and  train_truth is not None and test_truth is not None  and test_data is None')
        dm = DenMune(train_data=train_data, train_truth=train_truth, test_truth=test_truth, test_data=None, k_nearest=10)
        labels, validity = dm.fit_predict() 


    

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
    

# check if data will be treated correctly when comes as dataframe
def test_dataframe():
    data_file = 'https://raw.githubusercontent.com/egy1st/datasets/dd90854f92cb5ef73b4146606c1c158c32e69b94/denmune/shapes/aggr_rand.csv'
    data = pd.read_csv(data_file, sep=',', header=None)
    labels = data.iloc[:, -1]
    data = data.drop(data.columns[-1], axis=1)

    train_data = data [:555]
    test_data = data [555:]
    train_labels = labels [:555]
    test_labels = labels [555:]
    
    knn = 11 # k-nearest neighbor, the only parameter required by the algorithm
    dm = DenMune(train_data=train_data, train_truth=train_labels, test_data=test_data, test_truth=test_labels, k_nearest=knn, rgn_tsne=True)
    labels, validity = dm.fit_predict(validate=True, show_noise=True, show_analyzer=True)
    assert ( np.mean(dm.labels_pred == labels) > 0.97 or (1 - np.mean( dm.labels_pred == labels)  > 0.97) ) 