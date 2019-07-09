import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from yellowbrick.classifier import ConfusionMatrix


credito = pd.read_csv('Credit.csv')

#deve ser feita uma variavel para todas variaveis explicativas e uma para a var
# resposta, no caso para saber se o cliente é bom ou mau pagador

previsores = credito.iloc[:,0:20].values #tentar abrir o dataframe no spyder
# da erro , entao deve ser didigtado o indice diretamente no console: 
# ex: previsores[0] e classe

classe = credito.iloc[:,20].values #pega a coluna 20. na var previsores, como o
#intervalo vai de 0:20, vai ate a coluna 19

#no dataframe existem atributos em string. O algoritmo GaussianNB nao trabalha com
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
previsores[:,19] = labelencoder.fit_transform(previsores[:,19])


#treinamento

#deve ser feita a divisão da base de dados. O test_size diz a quantidade de dados
#que sera usada no teste, no caso 30%, enquanto os 70% restantes serão treinados
# o random_state=0 é pra dividir os dados sempre da mesma maneira
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                 classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
#aplicacao do treinamento. Gera a tabela de probabilidade
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

#aplicacao para clientes novos
novo_credito = pd.read_csv('NovoCredit.csv')
novo_credito = novo_credito.iloc[:, 0:20].values #transforma o formato em numpy array
#os dados devem ser os mesmos do treinamento, e na mesma ordem
novo_credito[:,0] = labelencoder.fit_transform(novo_credito[:,0])
novo_credito[:,2] = labelencoder.fit_transform(novo_credito[:,2])
novo_credito[:,3] = labelencoder.fit_transform(novo_credito[:,3])
novo_credito[:,5] = labelencoder.fit_transform(novo_credito[:,5])
novo_credito[:,6] = labelencoder.fit_transform(novo_credito[:,6])
novo_credito[:,8] = labelencoder.fit_transform(novo_credito[:,8])
novo_credito[:,9] = labelencoder.fit_transform(novo_credito[:,9])
novo_credito[:,11] = labelencoder.fit_transform(novo_credito[:,11])
novo_credito[:,13] = labelencoder.fit_transform(novo_credito[:,13])
novo_credito[:,14] = labelencoder.fit_transform(novo_credito[:,13])
novo_credito[:,16] = labelencoder.fit_transform(novo_credito[:,16])
novo_credito[:,18] = labelencoder.fit_transform(novo_credito[:,18])
novo_credito[:,19] = labelencoder.fit_transform(novo_credito[:,19])

naive_bayes.predict(novo_credito)







