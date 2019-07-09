#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:26:56 2019

@author: rudygualandro
"""

#serÃ¡ feito o algoritmo com todos os atributos e depois com alguns atributos
#selecionados para comparar os resultados

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

credito = pd.read_csv('Credito.csv', sep = ';', encoding = 'cp860')

#deve ser feita uma variavel para todas variaveis explicativas e uma para a var
# resposta, no caso para saber se o cliente Ã© bom ou mau pagador

previsores = credito.iloc[:,0:19].values #tentar abrir o dataframe no spyder
# da erro , entao deve ser didigtado o indice diretamente no console: 
# ex: previsores[0] e classe

classe = credito.iloc[:,19].values #pega a coluna 20. na var previsores, como o
#intervalo vai de 0:20, vai ate a coluna 19

#no dataframe existem atributos em string. O algoritmo nao trabalha com
# string, entao deve ser feita conversao. Os atributos categoricos devem ser subs-
#tituidos por valor numÃ©rico, em cada coluna:
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder.fit_transform(previsores[:,6])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder.fit_transform(previsores[:,9])
previsores[:,11] = labelencoder.fit_transform(previsores[:,11])
previsores[:,13] = labelencoder.fit_transform(previsores[:,13])
previsores[:,14] = labelencoder.fit_transform(previsores[:,14])
previsores[:,16] = labelencoder.fit_transform(previsores[:,16])
previsores[:,18] = labelencoder.fit_transform(previsores[:,18])



#treinamento

#deve ser feita a divisÃ£o da base de dados. O test_size diz a quantidade de dados
#que sera usada no teste, no caso 30%, enquanto os 70% restantes serÃ£o treinados
# o random_state=0 Ã© pra dividir os dados sempre da mesma maneira
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                 classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
svm = SVC()
svm.fit(X_treinamento, y_treinamento)
previsoes = svm.predict(X_teste)
taxa_acerto = accuracy_score(y_teste, previsoes)

#o algoritmo vai selecionar os atributos mais importantes
forest = ExtraTreesClassifier() #algoritmo de floresta randomica
forest.fit(X_treinamento, y_treinamento)
importancias = forest.feature_importances_ #nesta var sera criada lista dando
#uma pontuacao de importancia para cada atributo

#agora vao ser usados alguns atributos mais importantes no algoritmo

X_treinamento2 = X_treinamento[:,[0,1,2]]
X_teste2 = X_teste[:, [0,1,2]]

svm2 = SVC()
svm2.fit(X_treinamento2, y_treinamento)
previsoes2 = svm2.predict(X_teste2)
taxa_acerto2 = accuracy_score(y_teste, previsoes2)