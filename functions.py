from sklearn.metrics import confusion_matrix
import os.path
from sklearn import metrics
import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import time
from sklearn.manifold import TSNE

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.99, 's' : 80, 'linewidths':0}


class Functions():
               
        
    def plot_clusters(data, labels, alg_name, dp_name):
        
        palette = sns.color_palette('deep', np.unique(labels).max() + 1) #deep, dark, bright, muted, pastel, colorblind
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.show()
        plt.clf()    # this is a must to clear figures if you plot continously

        return 0
    
    def generate_tsne(f_name, k):
        
        filename = f_name
        filename2 = 'data/' + filename + '-2d'
        file_to_save = filename2 + ".txt"

        data = genfromtxt('data/' + filename + '.txt' , delimiter='\t')
        dim_two = TSNE(n_components=k).fit_transform(data)

        mystr = ""
        data_len = len (dim_two)
        for i in range(data_len):
            for n in range(k):
                mystr += str(round(dim_two[i][n],6))
                if (n  < k-1):mystr += '\t'
                if (n  == k-1): mystr += '\n'

        file_to_save = filename2 + ".txt"
        text_file = open(file_to_save, "w")
        text_file.write(mystr)
        text_file.close()

        return file_to_save

    def match_Labels(labels_pred, labels_true):
       
        list_pred = labels_pred.tolist()
        pred_set = set(list_pred) 

        index = []
        x = 1
        old_item = labels_true[0]
        old_x = 0

        for item in labels_true:

            if item != old_item:
                count = x - old_x
                index.append([old_x, old_item, count])
                old_item = item
                old_x = x
            x+= 1    

        ln = len(labels_true)
        count = x - old_x
        index.append([old_x, old_item, count])
        index[0][2] = index[0][2] -1

        index.sort(key=lambda x: x[2], reverse=True)

        lebeled = []
        for n in range (len(index)):
            newval = index[n][1]
            max_class = max(set(list_pred), key = list_pred[index[n][0]:index[n][0]+index[n][2]-1].count)
            if max_class not in lebeled:
                list_pred = [newval if x==max_class else x for x in list_pred]
                lebeled.append(newval)

        list_pred = np.array(list_pred)
        list_pred = list_pred.astype(np.int64)

        return list_pred
    
 