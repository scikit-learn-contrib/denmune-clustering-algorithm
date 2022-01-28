DenMune: A density-peak clustering algorithm
=============================================

DenMune a clustering algorithm that can find clusters of arbitrary size, shapes and densities in two-dimensions. Higher dimensions are first reduced to 2-D using the t-sne. The algorithm relies on a single parameter K (the number of nearest neighbors). The results show the superiority of the algorithm. Enjoy the simplicity but the power of DenMune.


[![PyPI Version](https://img.shields.io/pypi/v/denmune.svg)]( https://pypi.org/project/denmune/)
[![Launch notebook examples in Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD)
[![read the documentation](https://img.shields.io/badge/read_the-docs-orange)](https://docs.zerobytes.one/denmune/)
[![Launch notebook examples in Colaboratory, Google Research]( https://colab.research.google.com/assets/colab-badge.svg)](#colab)
[![Launch notebook examples in Kaggle, the workspace where data scientist meet](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/denmune-clustering-iris-dataset?scriptVersionId=84775816)
[![Elsevier, journal's article publisher ](https://img.shields.io/badge/elsevier-published-orange)](https://www.sciencedirect.com/science/article/abs/pii/S0031320320303927)
[![Research datasets at  Mendeley ](https://img.shields.io/badge/mendeley-data-bluegreen)](https://data.mendeley.com/datasets/b73cw5n43r/4)
[![BSD 3-Clause “New” or “Revised” License" ](https://img.shields.io/badge/license-BSD-green)](https://choosealicense.com/licenses/bsd-3-clause/)
[![CircleCI, continuous integration](https://circleci.com/gh/egy1st/denmune-clustering-algorithm/tree/main.svg?style=svg)](https://circleci.com/gh/egy1st/denmune-clustering-algorithm/tree/main)

Based on the paper
-------------------

|Paper|Journal|      
|-------------------------------------------------------------------------------------------|-----------------------------|
|Mohamed Abbas, Adel El-Zoghabi, Amin Ahoukry,                                              <td rowspan="5">[![scimagojr](https://www.scimagojr.com/journal_img.php?id=24823)](https://www.scimagojr.com/journalsearch.php?q=24823&tip=sid&clean=0)   </td>                           
|*DenMune: Density peak based clustering using mutual nearest neighbors*                                           
|In: Journal of Pattern Recognition, Elsevier,                                                                 
|volume 109, number 107589, January 2021                                                                      
|DOI: https://doi.org/10.1016/j.patcog.2020.107589                                                 

Documentation:
---------------
   Documentation, including tutorials, are available on https://docs.zerobytes.one/denmune
   
   [![read the documentation](https://img.shields.io/badge/read_the-docs-orange)](https://docs.zerobytes.one/denmune/)
   
 
Watch it in action
-------------------
This 30 seconds will tell you how a density-baased algorithm, DenMune propagates:

[![Propagation in DenMune](https://raw.githubusercontent.com/egy1st/denmune-clustering-algorithm/main/images/propagation.gif)]()

How to install DenMune
------------------------
Simply install DenMune clustering algorithm using pip command from the official Python repository

[![PyPI Version](https://img.shields.io/pypi/v/denmune.svg)]( https://pypi.org/project/denmune/)

From the shell run the command

```shell
pip install denmune
```

From jupyter notebook cell run the command

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



Interact with the algorithm
---------------------------
[![chameleon datasets](https://raw.githubusercontent.com/egy1st/denmune-clustering-algorithm/main/images/chameleon_detection.png)](https://colab.research.google.com/drive/1EUROd6TRwxW3A_XD3KTxL8miL2ias4Ue?usp=sharing)

This notebook allows you interact with the algorithm in many asspects:
- you can choose which dataset to cluster (among 4 chameleon datasets)
- you can decide which number of k-nearest neighbor to use 
- show noise on/off; thus you can invesitigate noise detected by the algorithm
- show analyzer on/off

How to run and test
--------------------

1. Launch Examples in Repo2Dpcker Binder

   Simply use our repo2docker offered by mybinder.org, which encapsulate the algorithm and all required data in one virtual machine instance. All jupter notebooks examples found in this repository will be also available to you in action to practice in this respo2docer. Thanks mybinder.org, you made it possible!

   [![Launch notebook examples in Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD)

2. Launch each Example in Kaggle workspace

   If you are a kaggler like me, then Kaggle, the best workspace where data scientist meet, should fit you to test the algorithm with great experince. 
   
   | Dataset | Kaggle URL |
   ----------| ---------------------------------------------------------------------------------------------------|
   | Non-groundtruth datasets |[![Non-groundtruth datasets](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/detecting-non-groundtruth-datasets) |
   | 2D Shape datasets |[![2D Shapes dataset](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/detection-of-2d-shape-datasets) |
   | MNIST dataset |[![MNIST dataset](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/get-97-using-simple-yet-one-parameter-algorithm) |
   |Iris dataset | [![iris dataset](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/denmune-clustering-iris-dataset) |
   |The beayty of propagation | [![The beayty of propagation](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/egyfirst/the-beauty-of-clusters-propagation) |
  
3. Launch each Example in Google Research, CoLab

   Need to test examples one by one, then here another option. Use colab offered by google research to test each example individually.
  

   <a name="colab"></a>
   Here is a list of Google CoLab URL to use the algorithm interactively
   ----------------------------------------------------------------------


   | Dataset | CoLab URL |
   ----------| ---------------------------------------------------------------------------------------------------|
   | Chameleon datasets |[![Chameleon dataset]( https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EUROd6TRwxW3A_XD3KTxL8miL2ias4Ue?usp=sharing) |
   | 2D Shape datasets |[![2D Shapes dataset]( https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1EaqTPCRHSuTKB-qEbnWHpGKFj6XytMIk?usp=sharing) |
   | MNIST dataset |[![MNIST dataset](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a9FGHRA6IPc5jhLOV46iEbpUeQXptSJp?usp=sharing) |
   | Non-groundtruth datasets |[![Non-groundtruth datasets](https://colab.research.google.com/assets/colab-badge.svg)]( https://colab.research.google.com/drive/1d17ejQ83aUy0CZIeQ7bHTugSC9AjJ2mU?usp=sharing) |
  


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
- [ ] create conda package

