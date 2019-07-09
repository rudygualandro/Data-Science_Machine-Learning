#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:41:47 2019

@author: rudygualandro
"""

from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
import skfuzzy

#diferentemente do Kmeans, que indica o grupo especifico, este algoritmo indica 
#a probabilidade de cada item se enquadrar em cada grupo(o item pode ser associado
#a mais de um grupo)


iris=datasets.load_iris()

r = skfuzzy.cmeans(data = iris.data.T, c = 3, m=2, error = 0.005,
                   maxiter = 1000, init = None)

#o data.T transpoe a matrix, oq eh necessario para usar este algoritmo
#O c=3 eh o numero de clusters
#o m=2 é membership, indica a qual grupo cada cluster pertence 
#sendo o parametro 2 recomendado pela documentacao
#o error tem a quantidade indicada pela documentacao, é um criterio de parada
#o maxiter é a quantidade de repeticoes


previsoes_porcentagem = r[1]

previsoes_porcentagem[0][0]

previsoes_porcentagem[1][0]

previsoes_porcentagem[2][0]

previsoes = previsoes_porcentagem.argmax(axis=0)#pega a maior de um item pertencer
#a um grupo e atribui esse item ao grupo correspondente 

resultados = confusion_matrix(iris.target, previsoes)
