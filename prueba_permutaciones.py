# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import itertools
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from data_processing import training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

clf_mlp=[[(20),(40),(55),(10,10),(20,20),(40,40),(55,55)],
['identity','logistic'],['constant', 'invscaling', 'adaptive'],
[0.001,0.01,0.1,0.2],[500,1000,2000]]

# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
clf_Svm=[[0.1,0.5,1],['linear', 'sigmoid']]

clf_Knn=[[2,3,4],['uniform', 'distance']]


clf_adaNB=[[GaussianNB()],[5,10,20,15,40,50],[0.001,0.01,0.1,0.5,0.7]]
arbol=[DecisionTreeClassifier(min_samples_leaf=3),DecisionTreeClassifier(min_samples_leaf=5),DecisionTreeClassifier(min_samples_leaf=7)]
clf_adaTree=[[arbol[0],arbol[1],arbol[2]],[5,10,20,15,40,50],[0.001,0.01,0.1,0.5,0.7]]
# adaBoost1 =AdaBoostClassifier(NB,n_estimators=50,learning_rate=0.1, 
#                                                       random_state=0)
# adaBoost2 =AdaBoostClassifier(tree,n_estimators=50,learning_rate=0.1, 
#                                                       random_state=0)


def best_hyperparameter(DataPartitions,YPartitions,key):
    print('MLP')
    score_vector=[]
    creado=[]
    permutaciones = []
    if(key=="MLP"):
        permutaciones = list(itertools.product(clf_mlp[0],clf_mlp[1],clf_mlp[2],clf_mlp[3],clf_mlp[4]))
        for i in range(len(permutaciones)):
            # print(f"Permutacion{i}")
            mlp = MLPClassifier(hidden_layer_sizes=permutaciones[i][0],activation=permutaciones[i][1],
                    learning_rate=permutaciones[i][2], learning_rate_init=permutaciones[i][3], 
                    random_state=1,max_iter=permutaciones[i][4])    
            creado.append(mlp)
            score,_=training(DataPartitions,YPartitions,mlp)
            score_vector.append(score)
            # print("SCORE:",score[i])
        indx_max=np.argmax(score_vector)
        print("MAYOR INDX:",indx_max)
    elif(key=="SVN"):
        print('SVN')
        permutaciones = list(itertools.product(clf_Svm[0],clf_Svm[1]))
        for i in range(len(permutaciones)):
            # print(f"Permutacion{i}")
            svm = SVC(C=permutaciones[i][0], kernel=permutaciones[i][1], random_state=0) 
            creado.append(svm)
            score,_=training(DataPartitions,YPartitions,svm)
            score_vector.append(score)
            # print("SCORE:",score[i])
        indx_max=np.argmax(score_vector)
        print("MAYOR INDX:",indx_max)
    elif(key=="KNN"):
        print('KNN')
        permutaciones = list(itertools.product(clf_Knn[0],clf_Knn[1]))
        for i in range(len(permutaciones)):
            # print(f"Permutacion{i}")
            knn = KNeighborsClassifier(n_neighbors=permutaciones[i][0], weights=permutaciones[i][1])
            creado.append(knn)
            score,_=training(DataPartitions,YPartitions,knn)
            score_vector.append(score)
            # print("SCORE:",score[i])
        indx_max=np.argmax(score_vector)
        print("MAYOR INDX:",indx_max)    
    elif(key=="AdaBoostNB"):
        permutaciones = list(itertools.product(clf_adaNB[0],clf_adaNB[1],clf_adaNB[2]))
        for i in range(len(permutaciones)):
            # print(f"Permutacion{i}")
            adaBoost1 = AdaBoostClassifier(permutaciones[i][0],n_estimators=permutaciones[i][1],learning_rate=permutaciones[i][2],random_state=0)  
            creado.append(adaBoost1)
            score,_=training(DataPartitions,YPartitions,adaBoost1)
            score_vector.append(score)
            # print("SCORE:",score[i])
        indx_max=np.argmax(score_vector)
        print("MAYOR INDX:",indx_max)
    elif(key=="AdaBoostTree"):
        permutaciones = list(itertools.product(clf_adaTree[0],clf_adaTree[1],clf_adaTree[2]))
        for i in range(len(permutaciones)):
            # print(f"Permutacion{i}")
            adaBoost2 = AdaBoostClassifier(permutaciones[i][0],n_estimators=permutaciones[i][1],learning_rate=permutaciones[i][2],random_state=0)  
            creado.append(adaBoost2)
            score,_=training(DataPartitions,YPartitions,adaBoost2)
            score_vector.append(score)
            # print("SCORE:",score[i])
        indx_max=np.argmax(score_vector)
        print("MAYOR INDX:",indx_max)
        
    return permutaciones[indx_max],creado[indx_max],score_vector[indx_max]