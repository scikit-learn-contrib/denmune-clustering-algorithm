=============================================
DenMune: A density-peak clustering algorithm
=============================================

DenMune a clustering algorithm that can find clusters of arbitrary size, shapes and densities in two-dimensions. Higher dimensions are first reduced to 2-D using the t-sne. The algorithm relies on a single parameter K (the number of nearest neighbors). The results show the superiority of the algorithm. Enjoy the simplicity but the power of DenMune.

.. image:: https://img.shields.io/pypi/v/denmune.svg
    :target: https://pypi.org/project/denmune/
    :alt: PyPI Version
    
.. image:: https://static.mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD
    :alt: Launch example notebooks in Binder


.. image:: https://readthedocs.org/projects/denmune-docs/badge
    :target: https://denmune-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
       
.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: #colab
    :alt: Launch example notebooks in Colaboratory, Google Research
    
.. image:: https://img.shields.io/badge/elsevier-published-orange
    :target: https://www.sciencedirect.com/science/article/abs/pii/S0031320320303927
    :alt: Research datasets at  Mendeley    
           
.. image:: https://img.shields.io/badge/mendeley-data-bluegreen
    :target: https://data.mendeley.com/datasets/b73cw5n43r/4
    :alt: Elsevier, journal's article publisher  
    
.. image:: https://img.shields.io/badge/license-BSD-green
    :target: https://choosealicense.com/licenses/bsd-3-clause/
    :alt: BSD 3-Clause “New” or “Revised” License"   
    
.. image:: https://circleci.com/gh/egy1st/denmune-clustering-algorithm/tree/main.svg?style=svg
    :target: https://circleci.com/gh/egy1st/denmune-clustering-algorithm/tree/main
    :alt: CircleCI, continuous integration     
     
Based on the paper
-------------------

    Mohamed Abbas, Adel El-Zoghabi, Amin Ahoukry, *DenMune: Density peak based clustering using mutual nearest neighbors*
    In: Journal of Pattern Recognition, Elsevier, volume 109, number 107589, January 2021
    
    DOI: https://doi.org/10.1016/j.patcog.2020.107589
    
 
Documentation:
---------------
   Documentation, including tutorials, are available on ReadTheDocs at https://denmune-docs.readthedocs.io/en/latest/. 
   
   .. image:: https://readthedocs.org/projects/denmune-docs/badge
    :target: https://denmune-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
 
Watch it in action
-------------------
    This 30 seconds will tell you how a density-baased algorithm, DenMune propagates
    
.. image:: https://github.com/egy1st/denmune-clustering-algorithm/blob/main/images/denmune_propagation.png
    :target: https://player.vimeo.com/video/663107261?h=08270149a9
    :alt: Propagation in DenMune      
    

    [![Propagation in DenMune]()]( "Propagation in DenMune")


    
   



 
 


How to install DenMune
--------------------------

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
----------------------
after installing DenMune, you just need to import it 

```python
from denmune import DenMune
```


------
Citing
------

If you have used this codebase in a scientific publication and wish to cite it, please use the `Journal of Pattern Recognition article <https://www.sciencedirect.com/science/article/abs/pii/S0031320320303927>`_.

    Mohamed Abbas McInnes, Adel El-Zoghaby, Amin Ahoukry, *DenMune: Density peak based clustering using mutual nearest neighbors*
    In: Journal of Pattern Recognition, Elsevier, volume 109, number 107589.
    January 2021
    
.. code:: bibtex

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
        keywords = {Clustering, Mutual neighbors, Dimensionality reduction, Arbitrary shapes, Pattern recognition, Nearest neighbors, Density peak}
      }
   

------------
Licensing
------------

The DenMune algorithm is 3-clause BSD licensed. Enjoy.

.. image:: https://img.shields.io/badge/license-BSD-green
    :target: https://choosealicense.com/licenses/bsd-3-clause/
    :alt: BSD 3-Clause “New” or “Revised” License"    
   


Task List
------------

- [x] Update Github with the DenMune sourcode
- [x] create repo2docker repository
- [x] Create pip Package
- [x] create colab shared examples
- [x] create documentation
- [ ] create conda package



