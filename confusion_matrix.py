# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:14:15 2023

@author: anita
"""

import matplotlib.pyplot as plt
from sklearn import metrics

def draw_confusion_matrix(real,predicted,clf,key):
    confusion_matrix = metrics.confusion_matrix(real, predicted,labels=[0,1])
    display_matrix= metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels=[0,1])
    display_matrix.plot(cmap=plt.cm.Blues)
    plt.title(f'{key}:Matriz de confusion')
    plt.savefig(f'{key}:matriz_de_confusión.png')
    plt.show()
