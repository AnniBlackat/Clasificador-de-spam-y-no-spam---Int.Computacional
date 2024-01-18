# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:10:17 2023

@author: anita
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from data_processing import k_partition,training, re_entrenamiento
from confusion_matrix import draw_confusion_matrix
from sklearn.metrics import f1_score,precision_score
from prueba_permutaciones import best_hyperparameter
# import pandas as pd

# =============================================================================
#                               CARGA DE DATOS
# =============================================================================
load_data=np.genfromtxt("dataset\spambase.data",delimiter=',')

data=load_data[:,:-1]
yd=load_data[:,-1:]
indxS=np.where(load_data[:,-1]==1)
print(indxS)
indxNS=np.where(load_data[:,-1]==0)
spam=data[indxS[0],:]
ham=data[indxNS[0],:]


print('spam', spam.shape)
print('ham', ham.shape)

#sobremuestreo la clase de menor muestras
remuestreo=resample(spam,replace=True,n_samples=ham.shape[0],random_state=42)
print(remuestreo.shape)
remuestreo=np.concatenate((remuestreo,np.ones((len(remuestreo),1))),axis=1)
print(remuestreo.shape)
ham=np.concatenate((ham,np.zeros((len(ham),1))),axis=1)
newData=np.concatenate((ham,remuestreo))

dataR=newData[:,:-1]
ydR=newData[:,-1:]

print("Data new:",dataR.shape)
print("Yd new:",ydR.shape)


# =============================================================================
#               GRAFICA DE DATOS DE ACUERDO A SU CLASIFICACIÓN
# =============================================================================
#Se grafica en funcion de su clasificación
x=np.arange(2)
y=np.array([list(ydR).count(0),list(ydR).count(1)])
labels=["Ham", "Spam"]
colors=["#90C2DE","#08306B"]
fig,ax=plt.subplots()
ax.set_title('Datos balanceados')
ax.bar(x,y,tick_label=labels,color=colors)
plt.savefig('img\datos_balanceados.png')
plt.show()

# =============================================================================
#               DIVISIÓN DE DATOS EN ENTRENAMIENTO Y PRUEBA
# =============================================================================

#Se dividen los datos en:
#80 --> para entrenamiento
#20 --> para pruebas
X_train, X_test, y_train, y_test = train_test_split(dataR, ydR, test_size=0.2, 
                                                random_state=42, shuffle=True, stratify=ydR)


indxS=np.where(y_train[:,-1]==1)

indxNS=np.where(y_train[:,-1]==0)
spamTrain=y_train[indxS[0],:]
hamTrain=y_train[indxNS[0],:]
print('spamTrain', spamTrain.shape)
print('hamTrain', hamTrain.shape)
indxS=np.where(y_test[:,-1]==1)

indxNS=np.where(y_test[:,-1]==0)
spamTest=y_test[indxNS[0],:]
hamTest=y_test[indxNS[0],:]
print('spamTest', spamTest.shape)
print('hamTest', hamTest.shape)
# =============================================================================
#             DIVISION DE DATOS DE ENTRENAMIENTO EN K PARTICIONES
# =============================================================================
#Se particiona el 80% de los datos de entrenamiento en k particiones
number_of_partitions=5
k_partition(X_train, y_train, number_of_partitions)

# =============================================================================
#                               CLASIFICADORES
# =============================================================================
# #Parámetros
# activacion="logistic"
# tasa_aprendizaje= [0.5, 0.1, 0.2]
# it_max=1000
# # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
# kType=["linear", "sigmoid"]
# k=[2,3,4]
# weight=['uniform', 'distance']
# arbol=[3,5,7]

# #Multicapa
# mlp = MLPClassifier(hidden_layer_sizes=(40),activation=activacion,
#                 learning_rate="constant", learning_rate_init=tasa_aprendizaje, 
#                 random_state=1,max_iter=it_max)
# #Maquina de vectores de soporte
# svm = SVC(C=0.1, kernel=kType, random_state=0)
# #Naives Bayes
# nb = GaussianNB()
# #KNN
# knn = KNeighborsClassifier(n_neighbors=k, weights=weight[0])
# #AdaBoost con Naives Bayes
# NB = GaussianNB()
# tree=DecisionTreeClassifier(min_samples_leaf=arbol)
# adaBoost1 =AdaBoostClassifier(NB,n_estimators=50,learning_rate=0.1, 
#                                                       random_state=0)
# adaBoost2 =AdaBoostClassifier(tree,n_estimators=50,learning_rate=0.1, 
#                                                       random_state=0)



# =============================================================================
#                       BUSQUEDA DE MEJORES HIPERPARAMETROS
# =============================================================================
besthpMlp,mlp,max_score=best_hyperparameter('partition\DataParticionadaTrain0.csv',
                      'partition\YParticionadaTrain0.csv',"MLP")

print(besthpMlp)
print(max_score)

besthpSvm,svm,max_score=best_hyperparameter('partition\DataParticionadaTrain0.csv',
                      'partition\YParticionadaTrain0.csv',"SVN")
print(besthpSvm)
print(max_score)

besthpKnn,knn,max_score=best_hyperparameter('partition\DataParticionadaTrain0.csv',
                      'partition\YParticionadaTrain0.csv',"KNN")
print(besthpKnn)
print(max_score)

besthpABNB,adaBoost1,max_score=best_hyperparameter('partition\DataParticionadaTrain0.csv',
                      'partition\YParticionadaTrain0.csv',"AdaBoostNB")
print(besthpABNB)
print(max_score)

besthpABT,adaBoost2,max_score=best_hyperparameter('partition\DataParticionadaTrain0.csv',
                      'partition\YParticionadaTrain0.csv',"AdaBoostTree")
print(besthpABT)
print(max_score)

#Naives Bayes
nb = GaussianNB()

#Multicapa
mlp = MLPClassifier(hidden_layer_sizes=besthpMlp[0],activation=besthpMlp[1],
                learning_rate=besthpMlp[2], learning_rate_init=besthpMlp[3], 
                random_state=1,max_iter=besthpMlp[4])
#Maquina de vectores de soporte
svm = SVC(C=besthpSvm[0], kernel=besthpSvm[1], random_state=0)
#Naives Bayes
nb = GaussianNB()
#KNN
knn = KNeighborsClassifier(n_neighbors=besthpKnn[0], weights=besthpKnn[1])
#AdaBoost con Naives Bayes
NB = GaussianNB()
adaBoost1 =AdaBoostClassifier(NB,n_estimators=besthpABNB[1],learning_rate=besthpABNB[2], 
                                                      random_state=0)

arbol=[DecisionTreeClassifier(min_samples_leaf=3),DecisionTreeClassifier(min_samples_leaf=5),DecisionTreeClassifier(min_samples_leaf=7)]
adaBoost2 =AdaBoostClassifier(besthpABT[0],n_estimators=besthpABT[1],learning_rate=besthpABT[2], 
                                                      random_state=0)

classifiers_dictionary={'MLP':mlp,'SVM':svm,'NaivesBayes':nb,'KNN':knn,
                        'AdaBoostNB':adaBoost1,'AdaBoostTree':adaBoost2}




score_vector=[]
precision_vector=[]
f1_score_vector=[]

# =============================================================================
#                               ENTRENAMIENTO
# =============================================================================

#Clasificadores a usar
for key, value in classifiers_dictionary.items():
    score=[]
    i=0

  #Ciclo para q vaya variando los parametros para cada modelo 
 
    while(i<number_of_partitions):
        score_p,clf=training(f'partition\DataParticionadaTrain{i}.csv',
                              f'partition\YParticionadaTrain{i}.csv',
                              value)
        # score_p,clf=process.entrenamiento(f'particion_{i+1}.csv',value)
        score.append(score_p)
        #Monitoreo que no se sobreentrene
        # if(len(score)>1 and score[i-1]>score[i]):
        #     y=i
        #     i=number_of_partitions
        i+=1
    print('-'*75)
    print(key)
    print('-'*75)
    print("Vector de score:",score)
    print("Media:",np.mean(score))
    print()
    
#reenttrenar
    while(i<number_of_partitions):
        score_pR,clfR=re_entrenamiento(f'partition\DataParticionadaTrain{i}.csv',
                              f'partition\YParticionadaTrain{i}.csv',
                              value)
# otra opción para no recorrer particion por particion 
#--> clf.fit(X_train, y_train) (fuera del while)
# # =============================================================================
# #                                PRUEBA
# # =============================================================================
    score_p=clf.score(X_test,y_test)
    print("Prueba:",score_p)
    print()
    y_predicted=clf.predict(X_test)           
    draw_confusion_matrix(y_test,y_predicted,clf,key) 
    f1_s=f1_score(y_test, y_predicted)
    precision=precision_score(y_test, y_predicted)
    f1_score_vector.append(f1_s)
    precision_vector.append(precision)
    score_vector.append(score_p)
    print(f"F1_Score {key}:",f1_s)
    print(f"Precision {key}:",precision)

classifiers = ['MLP', 'SVM','NB','KNN','ABNB','ABT']
# Se obtiene la posicion de cada etiqueta en x
x = np.arange(len(classifiers))
# Tamaño que tendrá cada barra
width = 0.3
fig, ax = plt.subplots()
colors=["#90C2DE","#08306B","#1C6BB0"]

rects1 = ax.bar(x - width, score_vector, width, label="Score",color=colors[0])
rects2 = ax.bar(x, f1_score_vector, width, label='F1',color=colors[1])
rects3 = ax.bar(x + width, precision_vector, width, label='Precision',
                    color=colors[2])

ax.set_title('Metricas')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.legend()
# plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1,step=0.1))
fig.tight_layout()
plt.savefig('img\metricas.png')
plt.show()
