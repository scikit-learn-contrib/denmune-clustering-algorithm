# DenMune: Density-Peak based Clustering Using Mutual Nearest Neighbors

This is the source code of DenMune Clustering Algorithm accompanied with the experimental work, which is published in Elsevier Pattern Recognition, Volume 109, January 2021, 107589
https://doi.org/10.1016/j.patcog.2020.107589

Authors: Mohamed Abbas, Adel El-Zoghabi, and Amin Shoukry

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD)

[![Documentation Status](https://readthedocs.org/projects/denmune-docs/badge/?version=latest)](https://denmune-docs.readthedocs.io/en/latest/?badge=latest)
      


Abstract
----

Many clustering algorithms fail when clusters are of arbitrary shapes, of varying densities, or the data classes are unbalanced and close to each other, even in two dimensions. A novel clustering algorithm “DenMune” is presented to meet this challenge. It is based on identifying dense regions using mutual nearest neighborhoods of size K, where K is the only parameter required from the user, besides obeying the mutual nearest neighbor consistency principle. The algorithm is stable for a wide range of values of K. Moreover, it is able to automatically detect and remove noise from the clustering process as well as detecting the target clusters. It produces robust results on various low and high dimensional datasets relative to several known state of the art clustering algorithms.

Installation
====
1- You may use it as docker using this 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD)

2- You may install it using pip using the command
<code>pip install denmune</code>
