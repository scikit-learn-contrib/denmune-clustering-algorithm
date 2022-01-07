DenMune a clustering algorithm that can find clusters of arbitrary size, shapes and densities in two-dimensions. Higher dimensions are first reduced to 2-D using the t-sne. The algorithm relies on a single parameter K (the number of nearest neighbors). The results show the superiority of DenMune. Enjoy the simplicity but the power of DenMune.

[![pypi repository](https://img.shields.io/pypi/v/denmune?logo=pypi "pypi repository")](https://pypi.org/project/denmune/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD)
[![Documentation Status](https://readthedocs.org/projects/denmune-docs/badge/?version=latest)](https://denmune-docs.readthedocs.io/en/latest/?badge=latest)
[![CircleCI](https://circleci.com/gh/egy1st/denmune-clustering-algorithm/tree/main.svg?style=svg)](https://circleci.com/gh/egy1st/denmune-clustering-algorithm/tree/main)
[![BSD 3-Clause “New” or “Revised” License](https://img.shields.io/badge/license-BSD-green "BSD 3-Clause “New” or “Revised” License")](https://choosealicense.com/licenses/bsd-3-clause/)
[![elsevier publisher](https://img.shields.io/badge/elsevier-published-orange "elsevier publisher")](https://www.sciencedirect.com/science/article/abs/pii/S0031320320303927)
[![mendeley data](https://img.shields.io/badge/mendeley-data-yellowgreen "mendeley data")](https://data.mendeley.com/datasets/b73cw5n43r/3)
[![interactive jupyter notebooks](https://img.shields.io/badge/notebook-interactive-brightgreen "interactive jupyter notebooks")](#colab)



This 30 seconds will tell you how a density-baased algorithm, DenMune propagates
===============
[![Propagation in DenMune](https://github.com/egy1st/denmune-clustering-algorithm/blob/main/images/denmune_propagation.png)](https://player.vimeo.com/video/663107261?h=08270149a9 "Propagation in DenMune")



Association
======
This code is associated with the research published in Elsvier Pattern Recognition Journal under DOI: https://doi.org/10.1016/j.patcog.2020.107589

How to install DenMune
====
Simply install DenMune clustering algorithm using pip command from the official Python repository

from the shell run the command
```shell
pip install denmune
```
from jupyter notebook cell run the command
```jupyter
!pip install denmune
```
How to use  DenMune
====
after installing DenMune, you just need to import it 

```python
from denmune import DenMune
```

Please note that first denmune (the package) in small letters, while the other one(the class itself) has D and M in capital case while other letters are small

How to run and test
======
Simply use our repo2docker offered by mybinder.org, which encapsulate the algorithm and all required data in one place and allow you to test over 11 examples. 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD)


Need to test examples one by one, then here another option. Use colab offered by google to test each example seperately.

<a name="colab"></a>
Here is a list of Google CoLab URL to use the algorithm interactively
----------------------------------------------------------------------


| Dataset | CoLab URL |
----------| ---------------------------------------------------------------------------------------------------|
| Aggregation dataset | [![Aggregation dataset](https://colab.research.google.com/assets/colab-badge.svg "Aggregation dataset")](https://colab.research.google.com/drive/1K-Uqp-fmETmic4VZoZvV5t5XgRTzf4KO?usp=sharing)
| Chameleon DS1 dataset | [![Chameleon DS1 dataset ](https://colab.research.google.com/assets/colab-badge.svg "Chameleon DS1 dataset ")](https://colab.research.google.com/drive/1LixPie1pZdWHxF1CXJIlwh1uTq-4iFYp?usp=sharing)
| Chameleon DS2 dataset | [![Chameleon DS2 dataset ](https://colab.research.google.com/assets/colab-badge.svg "Chameleon DS2 dataset ")](https://colab.research.google.com/drive/16Ve-1JJCgTQrX7ITJjDrSXWmwT9tG1AA?usp=sharing)
| Chameleon DS3 dataset | [![Chameleon DS3 dataset ](https://colab.research.google.com/assets/colab-badge.svg "Chameleon DS3 dataset ")](https://colab.research.google.com/drive/1mU5tV1sYWJpxqwyG-uA0yHMPZW7AzNuc?usp=sharing)
| Chameleon DS4 dataset | [![Chameleon DS4 dataset ](https://colab.research.google.com/assets/colab-badge.svg "Chameleon DS4 dataset ")]( https://colab.research.google.com/drive/1bDlsp1lVTDDXrDM8uWvo0_UY6ek73vUu?usp=sharing)



| Compound dataset | https://colab.research.google.com/drive/1TOv1mCLvAN24qvkh1f9H-ZERDgfoSMP6?usp=sharing |
| Iris dataset | https://colab.research.google.com/drive/1nKql57Xh7xVVu6NpTbg3vRdRg42R7hjm?usp=sharing |
| Jain dataset | https://colab.research.google.com/drive/1QJxXoZtoaMi3gvagZ2FPUtri4qbXOGl9?usp=sharing |
| Mouse dataset | https://colab.research.google.com/drive/11IpU1yaVaCa4H-d9yuwkjzywBfEfQGIp?usp=sharing |
| Pathbased dataset| https://colab.research.google.com/drive/17DofhHs5I2xyhnNPJ6RWETDf7Te71TKm?usp=sharing |
| Spiral dataset|https://colab.research.google.com/drive/1yW0Y14AiQYM6g7X4bJmUb3x3nson7Xup?usp=sharing |




Documentation
====
Rich documentation can be reached at:
[![Documentation Status](https://readthedocs.org/projects/denmune-docs/badge/?version=latest)](https://denmune-docs.readthedocs.io/en/latest/?badge=latest)




How to cite
=====
If you use DenMune code in scientific publications, we would appreciate citations.


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


Task List
====
- [x] Update Github with the DenMune sourcode
- [x] create repo2docker repository
- [x] Create pip Package
- [x] create colab shared examples
- [x] create documentation
- [ ] create conda package


