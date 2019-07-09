#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:19:45 2019

@author: rudygualandro
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.tree import export_graphviz

credito = pd.read_csv('Credito.csv', sep = ';', encoding = 'cp860')

#deve ser feita uma variavel para todas variaveis explicativas e uma para a var
# resposta, no caso para saber se o cliente é bom ou mau pagador

previsores = credito.iloc[:,0:19].values #tentar abrir o dataframe no spyder
# da erro , entao deve ser didigtado o indice diretamente no console: 
# ex: previsores[0] e classe

classe = credito.iloc[:,19].values #pega a coluna 20. na var previsores, como o
#intervalo vai de 0:20, vai ate a coluna 19

#no dataframe existem atributos em string. O algoritmo nao trabalha com
# string, entao deve ser feita conversao. Os atributos categoricos devem ser subs-
#tituidos por valor numérico, em cada coluna:
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

#deve ser feita a divisão da base de dados. O test_size diz a quantidade de dados
#que sera usada no teste, no caso 30%, enquanto os 70% restantes serão treinados
# o random_state=0 é pra dividir os dados sempre da mesma maneira
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                 classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

arvore = DecisionTreeClassifier()
arvore.fit(X_treinamento, y_treinamento)

#cria um arquivo .dot na pasta local. Deve ser copiado o conteudo do documento
# e colado no site do webgraphviz.com
export_graphviz(arvore, out_file = 'tree.dot') 

# para obter resultado, deve-se comparar a var previsoes com o y_teste
previsoes = arvore.predict(X_teste)
confusao = confusion_matrix(y_teste, previsoes)
taxa_acerto = accuracy_score(y_teste, previsoes)

