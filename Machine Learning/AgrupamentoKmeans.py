#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:13:41 2019

@author: rudygualandro
"""

from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris = datasets.load_iris()

unicos, quantidade = np.unique(iris.target, return_counts=True)#mostra em que grupo
#se enquadra cada item. Em seguida, será aplicado o KMeans para ver se ele agrupa
#os itens corretamente.

cluster = KMeans(n_clusters = 3)#é necessario definir qts clusters vc quer

cluster.fit(iris.data)

centroids = cluster.cluster_centers_#mostra os tres centros criados

previsoes = cluster.labels_#mostra em qual cluster cada item foi classificado

unicos2, quantidade2 = np.unique(previsoes, return_counts=True)#mostra a classificacao
#feita com a previsao

#faz-se a comparacao entre as variaveis quantidade e quantidade2 para ver o percentual
#de acerto do algoritmo

resultados = confusion_matrix(iris.target, previsoes)

plt.scatter(iris.data[previsoes==0, 0], iris.data[previsoes==0,1],
            c = 'green', label = 'Setosa')

plt.scatter(iris.data[previsoes==1, 0], iris.data[previsoes==1,1],
            c = 'red', label = 'Versicolor')

plt.scatter(iris.data[previsoes==2, 0], iris.data[previsoes==2,1],
            c = 'blue', label = 'Virginica')

plt.legend()