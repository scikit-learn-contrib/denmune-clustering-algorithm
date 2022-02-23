- [ ] DenMune: A density-peak clustering algorithm
  =============================================

  DenMune a clustering algorithm that can find clusters of arbitrary size, shapes and densities in two-dimensions. Higher dimensions are first reduced to 2-D using the t-sne. The algorithm relies on a single parameter K (the number of nearest neighbors). The results show the superiority of the algorithm. Enjoy the simplicity but the power of DenMune.


  Scientific Work
  ---------------------

| Paper                                                        | Journal                                                      | Data                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![Elsevier, journal's article publisher ](https://img.shields.io/badge/elsevier-published-orange)](https://www.sciencedirect.com/science/article/abs/pii/S0031320320303927) | [![scimagojr](https://www.scimagojr.com/journal_img.php?id=24823)](https://www.scimagojr.com/journalsearch.php?q=24823&tip=sid&clean=0) | [![Research datasets at  Mendeley ](https://img.shields.io/badge/mendeley-data-bluegreen)](https://data.mendeley.com/datasets/b73cw5n43r/4) |

  Coding & Maintenance
  -----------------------

| Git Repo                                                     | Code Style                                                   | Installation                                                 | CI Workflow                                                  | Code Coverage                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![GitHub commit activity](https://img.shields.io/github/commit-activity/m/egy1st/denmune-clustering-algorithm)](https://github.com/egy1st/denmune-clustering-algorithm) | ![Code Style: Black](https://img.shields.io/badge/code%20style-black-black) | [![PyPI Version](https://img.shields.io/pypi/v/denmune.svg)]( https://pypi.org/project/denmune/) | [![CircleCI, continuous integration](https://circleci.com/gh/egy1st/denmune-clustering-algorithm/tree/main.svg?style=shield)](https://circleci.com/gh/egy1st/denmune-clustering-algorithm/tree/main) | [![codecov](https://codecov.io/gh/egy1st/denmune-clustering-algorithm/branch/main/graph/badge.svg?token=E2ZY0DSUM2)](https://codecov.io/gh/egy1st/denmune-clustering-algorithm) |

  Docs & Tutorials
  ----------------------------

| Read the Docs                                                | Repo2Docker                                                  | Colab                                                        | kaggle                                                       | ReviewNB                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![Documentation Status](https://readthedocs.org/projects/denmune/badge/?version=latest)](https://denmune.readthedocs.io/en/latest/?badge=latest) | [![Launch notebook examples in Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD) | [![Launch notebook examples in Colaboratory, Google Research]( https://colab.research.google.com/assets/colab-badge.svg)](#colab) | [![Launch notebook examples in Kaggle, the workspace where data scientist meet](https://kaggle.com/static/images/open-in-kaggle.svg)](#kaggle) | [![Review NB](https://img.shields.io/badge/review-on%20reviewNB-blueviolet)](https://app.reviewnb.com/egy1st/denmune-clustering-algorithm/) |

  Downloads Stats
  --------------------

| downloads/day                                               | download/week                                               | download/month                                              |
| ----------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| ![PyPI - Downloads](https://img.shields.io/pypi/dd/denmune) | ![PyPI - Downloads](https://img.shields.io/pypi/dw/denmune) | ![PyPI - Downloads](https://img.shields.io/pypi/dm/denmune) |

  Suggestions & Reporting Issues
  -------------------------------

| Github                                                       | Gitter                                                       | Slack                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [![GitHub Discussions](https://img.shields.io/github/discussions/egy1st/denmune-clustering-algorithm)](https://github.com/egy1st/denmune-clustering-algorithm/discussions) | [![Gitter](https://badges.gitter.im/EG1ST/community.svg)](https://gitter.im/EG1ST/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) | [![Slack](https://img.shields.io/badge/chat-on%20slack-orange)](https://app.slack.com/client/T032DJEC8AW/C0326UK1VFY) |

  Based on the paper
  -------------------

  |Paper|
  |-------------------------------------------------------------------------------------------
  |Mohamed Abbas, Adel El-Zoghabi, Amin Shoukry,                                                                       
  |*DenMune: Density peak based clustering using mutual nearest neighbors*                                           
  |In: Journal of Pattern Recognition, Elsevier,                                                                 
  |volume 109, number 107589, January 2021                                                                      
  |DOI: https://doi.org/10.1016/j.patcog.2020.107589                                                 

  Documentation:
  ---------------

     Documentation, including tutorials, are available on https://denmune.readthedocs.io
    
     [![read the documentation](https://img.shields.io/badge/read_the-docs-orange)](https://denmune.readthedocs.io/en/latest/?badge=latest)


  Watch it in action
  -------------------

  This 30 seconds will tell you how a density-based algorithm, DenMune propagates:

  [![interact with the propagation](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o-tP3uvDGjxBOGYkir1lnbr74sZ06e0U?usp=sharing)

  [![Propagation in DenMune](https://raw.githubusercontent.com/egy1st/denmune-clustering-algorithm/main/images/propagation.gif)]()

  

  When less means more
  --------------------

  Most calssic clustering algorithms fail in detecting complex clusters where clusters are of different size, shape, density, and being exist in noisy data.
  Recently, a density-based algorithm named DenMune showed great ability in detecting complex shapes even in noisy data. it can detect number of clusters automatically, detect both pre-identified-noise and post-identified-noise automatically and removing them.

  It can achieve accuracy reach 100% in some classic pattern problems, achieve 97% in MNIST dataset. A great advantage of this algorithm is being single-parameter algorithm. All you need is to set number of k-nearest neighbor and the algorithm will care about the rest. Being Non-sensitive to changes in k, make it robust and stable.

  Keep in mind, the algorithm reduce any N-D dataset to only 2-D dataset initially, so it is a good benefit of this algorithm is being always to plot your data and explore it which make this algorithm a good candidate for data exploration. Finally, the algorithm comes with neat package for visualizing data, validating it and analyze the whole clustering process.

  How to install DenMune
  ------------------------

  Simply install DenMune clustering algorithm using pip command from the official Python repository

  [![PyPI Version](https://img.shields.io/pypi/v/denmune.svg)]( https://pypi.org/project/denmune/)

  From the shell run the command

  ```shell
pip install denmune
  ```

  From Jupyter notebook cell run the command

  ```ipython3
!pip install denmune
  ```

  How to use  DenMune
  --------------------

  Once DenMune is installed, you just need to import it 

  ```python
from denmune import DenMune
  ```

  ###### Please note that first denmune (the package) in small letters, while the other one(the class itself) has D and M in capital case.


  Read data
  -----------

  There are four possible cases of data:

  - only train data without labels
  - only labeled train data
  - labeled train data in addition to test data without labels
  - labeled train data in addition to labeled test data


  ```python
#=============================================
# First scenario: train data without labels 
# ============================================

data_path = 'datasets/denmune/chameleon/'  
dataset = "t7.10k.csv" 
data_file = data_path + dataset 

# train data without labels
X_train = pd.read_csv(data_file, sep=',', header=None)

knn = 39 # k-nearest neighbor, the only parameter required by the algorithm

dm = DenMune(train_data=X_train, k_nearest=knn)
labels, validity = dm.fit_predict(show_analyzer=False, show_noise=True)

  ```

  This is an intuitive dataset which has no groundtruth provided

  ![t710](https://raw.githubusercontent.com/egy1st/images/main/clustering/t710.png)

  ```python
#=============================================
# Second scenario: train data with labels 
# ============================================

data_path = 'datasets/denmune/shapes/'  
dataset = "aggregation.csv"
data_file = data_path + dataset 

# train data with labels
X_train = pd.read_csv(data_file, sep=',', header=None)
y_train = X_train.iloc[:, -1]
X_train = X_train.drop(X_train.columns[-1], axis=1)  

knn = 6 # k-nearest neighbor, the only parameter required by the algorithm

dm = DenMune(train_data=X_train, train_truth= y_train, k_nearest=knn)
labels, validity = dm.fit_predict(show_analyzer=False, show_noise=True)
  ```

  Datset groundtruth

  ![aggregation groundtruth](https://raw.githubusercontent.com/egy1st/images/main/clustering/aggregation_ground.png)

  Dataset as detected by DenMune at k=6

  ![aggregation train](https://raw.githubusercontent.com/egy1st/images/main/clustering/aggregation_6.png)


  ```python
#=================================================================
# Third scenario: train data with labels in addition to test data
# ================================================================

data_path = 'datasets/denmune/pendigits/'  
file_2d = data_path + 'pendigits-2d.csv'

# train data with labels 
X_train = pd.read_csv(data_path + 'train.csv', sep=',', header=None)
y_train = X_train.iloc[:, -1]
X_train = X_train.drop(X_train.columns[-1], axis=1)

# test data without labels
X_test = pd.read_csv(data_path + 'test.csv', sep=',', header=None) 
X_test = X_test.drop(X_test.columns[-1], axis=1)

knn = 50 # k-nearest neighbor, the only parameter required by the algorithm

dm = DenMune(train_data=X_train, train_truth= y_train,
             test_data= X_test, 
             k_nearest=knn)
labels, validity = dm.fit_predict(show_analyzer=True, show_noise=True)
  ```

  dataset groundtruth

  ![pendigits groundtruth](https://raw.githubusercontent.com/egy1st/images/main/clustering/pendigits_ground.png)


  dataset as detected by DenMune at k=50

  ![pendigits train](https://raw.githubusercontent.com/egy1st/images/main/clustering/pendigits_50.png)

  test data as predicted by DenMune on training the dataset at k=50

  ![pendigits test](https://raw.githubusercontent.com/egy1st/images/main/clustering/pendigits_test_50.png)


  Algorithm's Parameters
  -----------------------

    1. Parameters used within the initialization of the DenMune class

  ```python
def __init__ (self,
                  train_data=None, test_data=None,
                  train_truth=None, test_truth=None, 
                  file_2d =None, k_nearest=None, 
                  rgn_tsne=False, prop_step=0,
                  ):    
  ```

  - train_data:

    - data used for training the algorithm
    - default: None. It should be provided by the use, otherwise an error will raise.

  - train_truth:

    - labels of training data
    - default: None

  - test_data:

    - data used for testing the algorithm

  - test_truth:

    - labels of testing data
    - default: None

  - k_nearest:

    - number of nearest neighbor
    - default: 0. the default is invalid. k-nearest neighbor should be at least 1.

  - rgn_tsn: 

    - when set to True: It will regenerate the reduced 2-D version of the N-D dataset each time the algorithm run. 
    - when set to False: It will generate the reduced 2-D version of the N-D dataset first time only, then will reuse the saved exist file
    - default: True

  - file_2d: name (include location) of file used save/load the reduced 2-d version

    - if empty: the algorithm will create temporary file named '_temp_2d'
    - default: None

  - prop_step:

    - size of increment used in showing the clustering propagation.
    - leave this parameter set to 0, the default value, unless you are willing intentionally to enter the propagation mode.
    - default: 0

     

    2. Parameters used within the fit_predict function:

  ```python
 def fit_predict(self,
                    validate=True,
                    show_plots=True,
                    show_noise=True, 
                    show_analyzer=True
                    ):
  ```

  - validate:
    - validate data on/off according to five measures integrated with DenMune (Accuracy. F1-score, NMI index, AMI index, ARI index)
    - default: True

  - show_plots:
    - show/hide plotting of data
    - default: True

  - show_noise:
    - show/hide noise and outlier
    - default: True

  - show_analyzer:
    - show/hide the analyzer
    - default: True

  The Analyzer
  -------------

  The algorithm provide an exploratory tool called analyzer, once called it will provide you with in-depth analysis on how your clustering results perform.

  ![DenMune Analyzer](https://raw.githubusercontent.com/egy1st/images/main/clustering/analyzer.png)

  Noise Detection
  ----------------

  DenMune detects noise and outlier automatically, no need to any further work from your side.

  - It plots pre-identified noise in black
  - It plots post-identified noise in light grey

  You can set show_noise parameter to False.


  ```python
# let us show noise

m = DenMune(train_data=X_train, k_nearest=knn)
labels, validity = dm.fit_predict(show_noise=True)
  ```

  ```python
# let us show clean data by removing noise

m = DenMune(train_data=X_train, k_nearest=knn)
labels, validity = dm.fit_predict(show_noise=False)
  ```

| noisy data                                                   | clean data                                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![noisy data](https://raw.githubusercontent.com/egy1st/images/main/clustering/noisy_data.png) | ![clean data](https://raw.githubusercontent.com/egy1st/images/main/clustering/clean_data.png) |


  Validatation
  --------------

  You can get your validation results using 3 methods

  - by showing the Analyzer
  - extract values from the validity returned list from fit_predict function
  - extract values from the Analyzer dictionary
    -
      There are  five validity measures built-in the algorithm, which are:

  - ACC, Accuracy
  - F1 score
  - NMI index (Normalized Mutual Information)
  - AMI index (Adjusted Mutual Information)
  - ARI index (Adjusted Rand Index)

  ![Validation snapshot](https://raw.githubusercontent.com/egy1st/images/main/clustering/validation.png) 

  K-nearest Evolution
  -------------------

  The following chart shows the evolution of pre and post identified noise in correspondence to increase of number of knn. Also, detected number of clusters is analyzed in the same chart in relation with both types of identified  noise.

  ![knn evolution chart](https://raw.githubusercontent.com/egy1st/images/main/clustering/knn_vs_noise.png)


  The Scalability
  ----------------

| data size          | time                   |
| ------------------ | ---------------------- |
| data  size: 5000   | time: 2.3139 seconds   |
| data  size: 10000  | time: 5.8752 seconds   |
| data  size: 15000  | time: 12.4535 seconds  |
| data  size: 20000  | time: 18.8466 seconds  |
| data  size: 25000  | time: 28.992 seconds   |
| data  size: 30000  | time: 39.3166 seconds  |
| data  size: 35000  | time: 39.4842 seconds  |
| data  size: 40000  | time: 63.7649 seconds  |
| data  size: 45000  | time: 73.6828 seconds  |
| data  size: 50000  | time: 86.9194 seconds  |
| data  size: 55000  | time: 90.1077 seconds  |
| data  size: 60000  | time: 125.0228 seconds |
| data  size: 65000  | time: 149.1858 seconds |
| data  size: 70000  | time: 177.4184 seconds |
| data  size: 75000  | time: 204.0712 seconds |
| data  size: 80000  | time: 220.502 seconds  |
| data  size: 85000  | time: 251.7625 seconds |
| data  size: 100000 | time: 257.563 seconds  |

  | ![noisy data chart](https://raw.githubusercontent.com/egy1st/images/main/clustering/scalability.png)

  The Stability
  --------------

  The algorithm is only single-parameter, even more it not sensitive to changes in that parameter, k. You may guess that from the following chart yourself. This is of great benefit for you as a data exploration analyst. You can simply explore the dataset using an arbitrary k. Being Non-sensitive to changes in k, make it robust and stable.

  ![DenMune Stability chart](https://raw.githubusercontent.com/egy1st/images/main/clustering/stability.png)


  Reveal the propagation
  -----------------------

  one of the top performing feature in this algorithm is enabling you to watch how your clusters propagate to construct the final output clusters. just use the parameter 'prop_step' as in the following example:

  ```python
dataset = "t7.10k" #
data_path = 'datasets/denmune/chameleon/' 

# train file
data_file = data_path + dataset +'.csv'
X_train = pd.read_csv(data_file, sep=',', header=None)


from itertools import chain

# Denmune's Paramaters
knn = 39 # number of k-nearest neighbor, the only parameter required by the algorithm

# create list of differnt snapshots of the propagation
snapshots = chain(range(2,5), range(5,50,10), range(50, 100, 25), range(100,500,100), range(500,2000, 250), range(1000,5500, 500))

from IPython.display import clear_output
for snapshot in snapshots:
    print ("itration", snapshot )
    clear_output(wait=True)
    dm = DenMune(train_data=X_train, k_nearest=knn, rgn_tsne=False, prop_step=snapshot)
    labels, validity = dm.fit_predict(show_analyzer=False, show_noise=False)  
  ```

  [![Propagation in DenMune](https://raw.githubusercontent.com/egy1st/denmune-clustering-algorithm/main/images/propagation.gif)]()

  Interact with the algorithm
  ---------------------------

  [![chameleon datasets](https://raw.githubusercontent.com/egy1st/denmune-clustering-algorithm/main/images/chameleon_detection.png)](https://colab.research.google.com/drive/1EUROd6TRwxW3A_XD3KTxL8miL2ias4Ue?usp=sharing)

  This notebook allows you interact with the algorithm in many aspects:

  - you can choose which dataset to cluster (among 4 chameleon datasets)
  - you can decide which number of k-nearest neighbor to use 
  - show noise on/off; thus you can invesetigate noise detected by the algorithm
  - show analyzer on/off

  How to run and test
  --------------------

  1. Launch Examples in Repo2Docker Binder

     Simply use our repo2docker offered by mybinder.org, which encapsulate the algorithm and all required data in one virtual machine instance. All Jupyter notebooks examples found in this repository will be also available to you in action to practice in this respo2docer. Thanks mybinder.org, you made it possible!

     [![Launch notebook examples in Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD)

  <a name="colab"></a>

  2. Launch each Example in Google Research, CoLab
         
     Need to test examples one by one, then here another option. Use colab offered by google research to test each example individually.

     Here is a list of Google CoLab URL to use the algorithm interactively
     ----------------------------------------------------------------------


| Dataset                                          | CoLab URL                                                    |
| ------------------------------------------------ | ------------------------------------------------------------ |
| How to use it - colab                            | [![How to use it - colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1J_uKdhZ3z1KeY0-wJ7Ruw2PZSY1orKQm) |
| Chameleon datasets - colab                       | [![Chameleon datasets - colab]( https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EUROd6TRwxW3A_XD3KTxL8miL2ias4Ue?usp=sharing) |
| 2D Shape datasets - colab                        | [![2D Shape datasets - colab]( https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1EaqTPCRHSuTKB-qEbnWHpGKFj6XytMIk?usp=sharing) |
| MNIST dataset - colab                            | [![MNIST dataset - colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a9FGHRA6IPc5jhLOV46iEbpUeQXptSJp?usp=sharing) |
| iris dataset - colab                             | [![iris dataset - colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nKql57Xh7xVVu6NpTbg3vRdRg42R7hjm?usp=sharing) |
| Get 97% by training MNIST dataset - colab        | [![Get 97% by training MNIST dataset - colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1NeOtXEQY94oD98Ufbh3IhTHnnYwIA659) |
| Non-groundtruth datasets - colab                 | [![Non-groundtruth datasets - colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1d17ejQ83aUy0CZIeQ7bHTugSC9AjJ2mU?usp=sharing) |
| Noise detection - colab                          | [![Noise detection - colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1Bp3c-cJfjLWxupmrBJ_6Q4-nqIfZcII4) |
| Validation - colab                               | [![Validation - colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/13_EVaQOv_QiNmQiMWJAcFFHPJHGCrQLe) |
| How it propagates - colab                        | [![How it propagates - colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o-tP3uvDGjxBOGYkir1lnbr74sZ06e0U?usp=sharing) |
| Snapshots of propagation - colab                 | [![snapshots of the propagation - colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vPXNKa8Rf3TnqDHSD3YSWl3g1iNSqjl2?usp=sharing) |
| Scalability - colab                              | [![Scalability - colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1d55wkBndLLapO7Yx1ePHhE8mL61j9-TH?usp=sharing) |
| Stability vs number of nearest neighbors - colab | [![Stability vs number of nearest neighbors - colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17VgVRMFBWvkSIH1yA3tMl6UQ7Eu68K2l?usp=sharing) |
| k-nearest-evolution - colab                      | [![k-nearest-evolution - colab](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1DZ-CQPV3WwJSiaV3-rjwPwmXw4RUh8Qj) |

  <a name="kaggle"></a>  

  3. Launch each Example in Kaggle workspace

     If you are a kaggler like me, then Kaggle, the best workspace where data scientist meet, should fit you to test the algorithm with great experience. 


     | Dataset                                  | Kaggle URL                                                   |
     | ---------------------------------------- | ------------------------------------------------------------ |
     | When less means more - kaggle            | [![When less means more - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]( https://www.kaggle.com/egyfirst/when-less-means-more) |
     | Non-groundtruth datasets - kaggle        | [![Non-groundtruth datasets](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/detecting-non-groundtruth-datasets) |
     | 2D Shape datasets - kaggle               | [![2D Shape datasets - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/detection-of-2d-shape-datasets) |
     | MNIST dataset kaggle                     | [![MNIST dataset - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/get-97-using-simple-yet-one-parameter-algorithm) |
     | Iris dataset kaggle                      | [![iris dataset - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/denmune-clustering-iris-dataset) |
     | Training MNIST to get 97%                | [![Training MNIST to get 97%](https://kaggle.com/static/images/open-in-kaggle.svg)](  https://www.kaggle.com/egyfirst/training-mnist-dataset-to-get-97) |
     | Noise detection - kaggle                 | [![Noise detection - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]( https://www.kaggle.com/egyfirst/noise-detection) |
     | Validation - kaggle                      | [![Validation - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/validate-in-5-built-in-validity-insexes) |
     | The beauty of propagation - kaggle       | [![The beauty of propagation - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/the-beauty-of-clusters-propagation) |
     | The beauty of propagation part2 - kaggle | [![The beauty of propagation part 2 - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/the-beauty-of-propagation-part2) |
     | Snapshots of propagation -kaggle         | [![The beauty of propagation - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/beauty-of-propagation-part3) |
     | Scalability kaggle                       | [![Scalability - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/scalability-vs-speed) |
     | Stability - kaggle                       | [![Stability - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/stability-vs-number-of-nearest-neighbor) |
     | k-nearest-evolution - kaggle             | [![k-nearest-evolution - kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/k-nearest-evolution) |


  How to cite
  =====

  If you have used this codebase in a scientific publication and wish to cite it, please use the [Journal of Pattern Recognition article](https://www.sciencedirect.com/science/article/abs/pii/S0031320320303927)

      Mohamed Abbas McInnes, Adel El-Zoghaby, Amin Ahoukry, *DenMune: Density peak based clustering using mutual nearest neighbors*
      In: Journal of Pattern Recognition, Elsevier, volume 109, number 107589.
      January 2021


  ```bib
@article{ABBAS2021107589,
title = {DenMune: Density peak based clustering using mutual nearest neighbors},
journal = {Pattern Recognition},
volume = {109},
pages = {107589},
year = {2021},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2020.107589},
url = {https://www.sciencedirect.com/science/article/pii/S0031320320303927},
author = {Mohamed Abbas and Adel El-Zoghabi and Amin Shoukry},
keywords = {Clustering, Mutual neighbors, Dimensionality reduction, Arbitrary shapes, Pattern recognition, Nearest neighbors, Density peak},
abstract = {Many clustering algorithms fail when clusters are of arbitrary shapes, of varying densities, or the data classes are unbalanced and close to each other, even in two dimensions. A novel clustering algorithm “DenMune” is presented to meet this challenge. It is based on identifying dense regions using mutual nearest neighborhoods of size K, where K is the only parameter required from the user, besides obeying the mutual nearest neighbor consistency principle. The algorithm is stable for a wide range of values of K. Moreover, it is able to automatically detect and remove noise from the clustering process as well as detecting the target clusters. It produces robust results on various low and high dimensional datasets relative to several known state of the art clustering algorithms.}
}
  ```

  Licensing
  ------------

   The DenMune algorithm is 3-clause BSD licensed. Enjoy.

   [![BSD 3-Clause “New” or “Revised” License" ](https://img.shields.io/badge/license-BSD-green)](https://choosealicense.com/licenses/bsd-3-clause/)


  Task List
  ------------

  - [x] Update Github with the DenMune sourcode
  - [x] create repo2docker repository
  - [x] Create pip Package
  - [x] create CoLab shared examples
  - [x] create documentation
  - [x] create Kaggle shared examples
  - [x] PEP8 compliant
  - [x] Continuous integration
  - [x] scikit-learn compatible
  - [x] creating unit tests (coverage: 100%)
  - [x] generating API documentation
  - [ ] create conda package
