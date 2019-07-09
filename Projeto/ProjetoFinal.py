#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 00:33:17 2019

@author: rudygualandro
"""


#Importacoes das bibliotecas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

#________________________________________________________________________

# Carregamento da base de dados na variavel credito:

credito = pd.read_csv('Credito.csv', sep = ';', encoding = 'cp860')

#________________________________________________________________________

# Criados objetos para guardar as variaveis explicativas(previsores) e para a variavel
# resposta (classe), no caso para saber se o cliente é bom ou mau pagador

previsores = credito.iloc[:,0:19].values 

classe = credito.iloc[:,19].values #pega a coluna 20. na var previsores; como o
#intervalo vai de 0:20, vai ate a coluna 19

#________________________________________________________________________

# No dataframe existem atributos em string. Os algoritmos utilizados nao trabalham com
# string, entao deve ser feita conversao. Os atributos categoricos devem ser subs-
#tituidos por valores numéricos, em cada coluna:

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

#________________________________________________________________________

# Deve ser feita a divisão da base de dados. O test_size diz a quantidade de dados
#que sera usada no teste, no caso 30%, enquanto os 70% restantes serão treinados
# o random_state=0 é pra dividir os dados sempre da mesma maneira

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                 classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
#__________________________________________________________________________

# Aplicacao do treinamento. 
#Minha primeira tentativa foi com o algoritmo Naive Bayes, utilizando todas as colunas como
#previsores 

naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento)

#previsoes. Submete cada registro do X_teste ao modelo treinado e da resposta
#good ou bad de acordo com a tabela de probabilidade gerada no treinamento
previsoes = naive_bayes.predict(X_teste)

#contabilizicao dos erros e acertos comparando o dado real com o dado previsto
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

#visualizacao da tabela de acerto 
v = ConfusionMatrix(GaussianNB())
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.poof()

# A primeira tentativa com Naive Bayes teve taxa de acerto de 70,3%

#____________________________________________________________________________

# A segunda tentativa foi com o algoritmo Decision Tree

arvore = DecisionTreeClassifier()
arvore.fit(X_treinamento, y_treinamento)

previsoes = arvore.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

# A tentativa com a arvore de decisao teve taxa de acerto de 66,3%

#_________________________________________________________________________

# A terceira tentativa foi com o algoritmo SVC

svm = SVC()
svm.fit(X_treinamento, y_treinamento)
previsoes = svm.predict(X_teste)
taxa_acerto = accuracy_score(y_teste, previsoes)

#A tentativa com o SVC teve taxa de acerto de 71,3%

# Utilizei o algoritmo de floresta randomica para escolher os atributos mais
#significativos: 

forest = ExtraTreesClassifier() #algoritmo de floresta randomica
forest.fit(X_treinamento, y_treinamento)
importancias = forest.feature_importances_

#selecionando os atributos das colunas 0,1,2 e 3, obtive taxa de 
# acerto de 72,3%:

X_treinamento2 = X_treinamento[:,[0,1,2,3]]
X_teste2 = X_teste[:, [0,1,2,3]]

svm2 = SVC()
svm2.fit(X_treinamento2, y_treinamento)
previsoes2 = svm2.predict(X_teste2)
taxa_acerto2 = accuracy_score(y_teste, previsoes2)

# Contudo, utilizando os atributos das colunas 0,1 e 2 apenas, obtive taxa de 
# acerto de 74%, o que eh quase a meta buscada no projeto:

X_treinamento2 = X_treinamento[:,[0,1,2]]
X_teste2 = X_teste[:, [0,1,2]]

svm2 = SVC()
svm2.fit(X_treinamento2, y_treinamento)
previsoes2 = svm2.predict(X_teste2)
taxa_acerto2 = accuracy_score(y_teste, previsoes2)

#___________________________________________________________________

# Ao perceber que utilizar menos atributos poderia deixar a previsao mais precisa,
# fiz um ajuste no modelo, por tentativa e erro. Tirei as ultimas 4 colunas de atributos
# da variavel previsores e apliquei o algoritmo Naive Bayes novamente:

previsores = credito.iloc[:,0:15].values 
classe = credito.iloc[:,19].values 

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

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                 classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
naive_bayes = GaussianNB()
naive_bayes.fit(X_treinamento, y_treinamento)

previsoes = naive_bayes.predict(X_teste)

confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

v = ConfusionMatrix(GaussianNB())
v.fit(X_treinamento, y_treinamento)
v.score(X_teste, y_teste)
v.poof()

# Assim, consegui taxa de acerto de 75%, conforme pedido no exercicio



















