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
    
.. image:: https://kaggle.com/static/images/open-in-kaggle.svg
     :target: https://www.kaggle.com/egyfirst/denmune-clustering-iris-dataset?scriptVersionId=84775816\
     :alt: Launch example notebooks in Kaggle, the workspace where data scientist meet    
    
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

+-------------------------------------------------------------------------------------------+-----------------------------+        
|Mohamed Abbas, Adel El-Zoghabi, Amin Ahoukry,                                              |                             |
|*DenMune: Density peak based clustering using mutual nearest neighbors*                    |                             |
|In: Journal of Pattern Recognition, Elsevier,                                              |                             |
|volume 109, number 107589, January 2021                                                    |                             |
|DOI: https://doi.org/10.1016/j.patcog.2020.107589                                          | |scimagojr|                 | 
+-------------------------------------------------------------------------------------------+-----------------------------+

    
      
 
Documentation:
---------------
   Documentation, including tutorials, are available on ReadTheDocs at https://denmune-docs.readthedocs.io/en/latest/. 
   
   .. image:: https://readthedocs.org/projects/denmune-docs/badge
    :target: https://denmune-docs.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
 
Watch it in action
-------------------
    This 30 seconds will tell you how a density-baased algorithm, DenMune propagates

.. image:: https://raw.githubusercontent.com/egy1st/denmune-clustering-algorithm/main/images/denmune_propagation.png
 :target: https://player.vimeo.com/video/663107261?h=08270149a9
 :alt: Propagation in DenMune  

How to install DenMune
------------------------

    Simply install DenMune clustering algorithm using pip command from the official Python repository

.. image:: https://img.shields.io/pypi/v/denmune.svg
    :target: https://pypi.org/project/denmune/
    :alt: PyPI Version
    

from the shell run the command

    .. code:: python

        pip install denmune


from jupyter notebook cell run the command

    .. code:: jupyter

        !pip install denmune
        

How to use  DenMune
--------------------
Once DenMune is installed, you just need to import it 

    .. code:: python

        from denmune import DenMune
        # Please note that first denmune (the package) in small letters, while the other one(the class itself) has D and M in capital case.


--------------------
How to run and test
--------------------

#. Launch Examples in Repo2Dpcker Binder

    Simply use our repo2docker offered by mybinder.org, which encapsulate the algorithm and all required data in one virtual machine instance. All jupter notebooks examples found in this repository will be also available to you in action to practice in this respo2docer. Thanks mybinder.org, you made it possible!

    .. image:: https://static.mybinder.org/badge_logo.svg
        :target: https://mybinder.org/v2/gh/egy1st/denmune-clustering-algorithm/HEAD
        :alt: Launch example notebooks in Binder

#. Launch each Example in Google Research, CoLab

    Need to test examples one by one, then here another option. Use colab offered by google research to test each example individually.
  
    .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: #colab
     :alt: Launch example notebooks in Colaboratory, Google Research
     
#. Launch each Example in Kaggle workspace

    If you are a kaggler like me, then Kaggle, the best workspace where data scientist meet, should fit you to test the algorithm with great experince. (in progress ..........)
  
    .. image:: https://kaggle.com/static/images/open-in-kaggle.svg
     :target: https://www.kaggle.com/egyfirst/denmune-clustering-iris-dataset?scriptVersionId=84775816\
     :alt: Launch example notebooks in Kaggle, the workspace where data scientist meet

Here is a list of Google CoLab URL to use the algorithm interactively:


	+------------------------------+-------------------+        
	| Aggregation dataset          | |aggregation|     | 
	+------------------------------+-------------------+
	| Chameleon DS1 dataset        | |cham-ds1|        |
	+------------------------------+-------------------+     
	| Chameleon DS2 dataset        | |cham-ds2|        | 
	+------------------------------+-------------------+
	| Chameleon DS3 dataset        | |cham-ds3|        |
	+------------------------------+-------------------+
	| Chameleon DS4 dataset        | |cham-ds4|        |
	+------------------------------+-------------------+
	| Compound dataset             | |compound|        | 
	+------------------------------+-------------------+
	| iris dataset                 | |iris|            |
	+------------------------------+-------------------+     
	| Jian dataset                 | |jain|            | 
	+------------------------------+-------------------+
	| Mouse dataset                | |mouse|           |
	+------------------------------+-------------------+
	| Pathbased dataset            | |pathbased|       |
	+------------------------------+-------------------+
	| Spiral dataset               | |spiral|          |
	+------------------------------+-------------------+

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
- [x] create CoLab shared examples
- [x] create documentation
- [ ] create Kaggle shared examples
- [ ] create conda package


.. |aggregation| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1K-Uqp-fmETmic4VZoZvV5t5XgRTzf4KO?usp=sharing

.. |cham-ds1| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1LixPie1pZdWHxF1CXJIlwh1uTq-4iFYp?usp=sharing

.. |cham-ds2| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/16Ve-1JJCgTQrX7ITJjDrSXWmwT9tG1AA?usp=sharing

.. |cham-ds3| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1mU5tV1sYWJpxqwyG-uA0yHMPZW7AzNuc?usp=sharing

.. |cham-ds4| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1bDlsp1lVTDDXrDM8uWvo0_UY6ek73vUu?usp=sharing

.. |compound| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1TOv1mCLvAN24qvkh1f9H-ZERDgfoSMP6?usp=sharing
   
.. |iris| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1nKql57Xh7xVVu6NpTbg3vRdRg42R7hjm?usp=sharing
   
.. |jain| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1QJxXoZtoaMi3gvagZ2FPUtri4qbXOGl9?usp=sharing
      
.. |mouse| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/11IpU1yaVaCa4H-d9yuwkjzywBfEfQGIp?usp=sharing
   
.. |pathbased| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/17DofhHs5I2xyhnNPJ6RWETDf7Te71TKm?usp=sharing   
   
.. |spiral| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1yW0Y14AiQYM6g7X4bJmUb3x3nson7Xup?usp=sharing  

.. |scimagojr| image:: https://www.scimagojr.com/journal_img.php?id=24823
   :target: https://www.scimagojr.com/journalsearch.php?q=24823&tip=sid&clean=0
  
   
