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
               
      
    def backline(self):
        print('\r', end='') 
        
    def calc_Metrics(self, labels_true, labels_pred):
       
        # Score the clustering
        from sklearn.metrics.cluster import adjusted_mutual_info_score #2010
        from sklearn.metrics.cluster import adjusted_rand_score # 1985
        from sklearn.metrics import f1_score
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score

        #from sklearn.metrics import davies_bouldin_score 
        # #1975 - 2001    ## no ground truth   ##Values closer to zero indicate a better partition.

        from sklearn.metrics import pairwise_distances # for calinski_harabasz_score
        ## also known as the Variance Ratio Criterion - can be used to evaluate the model, 
        ## where a higher Calinski-Harabasz score relates to a model with better defined clusters.

        from sklearn import metrics # for homogeneity, completeness, fowlkes
        ##  homogeneity: each cluster contains only members of a single class.
        ## completeness: all members of a given class are assigned to the same cluster.
        #v-measure the harmonic mean of homogeneity and completeness called V-measure 2007

        acc = metrics.accuracy_score(labels_true, labels_pred, normalize=False)

        #mi = metrics.mutual_info_score(labels_true, labels_pred)
        #print("mutual_info_score: %f." %  mi)

        nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
        #print("normalized_mutual_info_score: %f." % nmi)

        ami = adjusted_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
            #print("Adjusted_mutual_info_score: %f." %  adj_nmi)

        homogeneity = metrics.homogeneity_score(labels_true, labels_pred)
        #print("homogeneity_score: %f." % homogeneity_score)

        completeness = metrics.completeness_score(labels_true, labels_pred)
        #print("completeness_score: %f." % completeness_score)

        f1_weight = metrics.f1_score(labels_true, labels_pred, average='weighted')
        #f1_micro = metrics.f1_score(labels_true, labels_pred, average='micro')
        #f1_macro = metrics.f1_score(labels_true, labels_pred, average='macro')
        #print("f1_score: %f." % f1_score)
        
        
    
        ari = adjusted_rand_score(labels_true, labels_pred)
        #print("adjusted_rand_score: %f." % adj_rand)

        

        f1 =  f1_weight
        val = ['0', acc, f1, nmi, ami, ari, homogeneity, completeness, '0' ]
         
        return val
    
    def match_Labels(self, labels_pred, labels_true):
       
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
    
       
        
   
    
    def plot_clusters(self, data, labels, alg_name, dp_name, show=False):
        
        palette = sns.color_palette('deep', np.unique(labels).max() + 1) #deep, dark, bright, muted, pastel, colorblind
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
        plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.savefig('results/' + alg_name +'/images/' + alg_name + '_' + dp_name +  '.png')
        
        if show == True:
            plt.show()
        
        plt.clf()    # this is a must to clear figures if you plot continously

        return 0
    
    def generate_tsne(self, f_name, k):
        
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
    
    
    def labels_Patterns (self, mylist):
        #mylist = [1, 26, 27, 51, 52, 77, 78, 103, 105]
        mylist = [str(i) for i in mylist]
        if  int(max (mylist)) <= 103:

            x = 0
            for item  in mylist:
                if item <= '25':
                    mylist[x] = chr(int(item)+66)
                elif item <= '51':
                    mylist[x] = 'A' + chr(int(item)+39)
                elif item <= '77':
                    mylist[x] = 'B' + chr(int(item)+13)
                else:
                    mylist[x] = 'C' + chr(int(item)-13)

                x += 1 
            return mylist    

        else:
            return 'Max classes numbers are 103 classes'
