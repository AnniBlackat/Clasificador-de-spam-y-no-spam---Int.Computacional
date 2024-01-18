# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 20:55:09 2023

@author: anita
"""
import numpy as np
import math as m
from sklearn.model_selection import train_test_split

#Se recibe el 80% de datos que se destino para entrenamiento
#Se realizan k particiones
def k_partition(data,y,cantParticiones):
    #Si queda un valor real y no entero se redondea
    #un numero hacia abajo al entero más cercano 
    cantDataPart=m.floor(len(data)/cantParticiones) 
    print(cantDataPart)
    #se realizan 5 particiones
    for i in range(cantParticiones):
        particiones=[]
        particionesYd=[]
        #Se separa tanto los datos como la salida conocida
        particiones = data[i*cantDataPart: (i+1)*cantDataPart, :]
        particionesYd =y[i*cantDataPart:(i+1)*cantDataPart] 
       
        np.savetxt(f'partition\DataParticionadaTrain{i}.csv', particiones, delimiter=",")
        np.savetxt(f'partition\YParticionadaTrain{i}.csv', particionesYd, delimiter=",")


def training(DataPartitions,YPartitions,clf):
       
    dataParticionada=np.genfromtxt(DataPartitions,delimiter=',')
    ydParticionada=np.genfromtxt(YPartitions,delimiter=',')
    #Se divide los datos en:
    #90%(sería un 70% general)--> para entrenamiento
    #10% --> para prueba
    X_train, X_test, y_train, y_test = train_test_split(dataParticionada,ydParticionada, 
                                            test_size=0.1, random_state=0, shuffle=True, stratify=ydParticionada)
    #Se entrena el modelo
    clf.fit(X_train, y_train)
    score=clf.score(X_test,y_test)
    # y = clf.predict(X_test)
    # print(classification_report(y_test,y))
    return score,clf

def re_entrenamiento(DataPartitions,YPartitions,clf):
       
    dataParticionada=np.genfromtxt(DataPartitions,delimiter=',')
    ydParticionada=np.genfromtxt(YPartitions,delimiter=',')
    #Se divide los datos en:
    #90%(sería un 70% general)--> para entrenamiento
    #10% --> para prueba
   
    #Se entrena el modelo
    clf.fit(dataParticionada, ydParticionada)
   


#OTRA ALTERNATIVA 
# def particionar(data,y,k):
#     particiones=[]
#     particionesYd=[]
#     #Cantidad de muestras
#     muestras=m.floor(len(data)/k)
#     #K particiones
#     for i in range(0,len(data),muestras):
#         particiones.append(np.array(data[i:i+muestras,:]))
#         particionesYd.append(np.array(y[i:i+muestras]))
#     return particiones,particionesYd

    
   