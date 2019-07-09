#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:05:40 2019

@author: rudygualandro
"""

from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer

#o algoritmo escolhe pontos reais na base de dados chamados medoids
#aqui serao usados apenas dois atributos para possibilitar a visualizacao
#iris.data[:,0:2]

iris = datasets.load_iris()

cluster = kmedoids(iris.data[:,0:2], [3,12,20] )#[3,12,20] sao indices aleatorios

cluster.get_medoids()

cluster.process()#faz o processamento

previsoes = cluster.get_clusters()#nao precisa configurar o numero de clusters
#o algoritmo determina o numero de clusters

medoides = cluster.get_medoids()#define os medoids, que sao os centros dos clusters

v = cluster_visualizer()

v.append_clusters(previsoes, iris.data[:,0:2])#sao passadas as previsoes e os
#dados reais

v.append_cluster(medoides, data = iris.data[:,0:2], marker='*', markersize=15)

v.show()

#deve ser feita codificacao para comparar o acerto do algoritmo, para que a var
#previsoes fique no mesmo padrao do target que esta dentro da var iris
#serao feitas duas listas alternando valores reais e previstos

lista_previsoes = []
lista_real = []

for i in range(len(previsoes)):
    
    print('______')
    print(i)
    print('______')
    for j in range(len(previsoes[i])):
        #print(j)
        print(previsoes[i][j])
        lista_previsoes.append(i)
        lista_real.append(iris.target[previsoes[i][j]])

#transformar as listas de valores reais e previstos em numpy array
lista_previsoes = np.asarray(lista_previsoes)        
lista_real = np.asarray(lista_real)  

resultados = confusion_matrix(lista_real, lista_previsoes)








