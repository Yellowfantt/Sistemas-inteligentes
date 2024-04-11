import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import CategoricalNB

df = pd.read_csv('jogar.csv')
df.head()

tempo_le = preprocessing.LabelEncoder()
tempo = tempo_le.fit_transform(df['tempo'].values)

print("Categorias encontradas para tempo: {}".format(tempo_le.classes_))
print("Exemplo de encode de tempo: {} e seu valor real: {}".format(tempo[0], tempo_le.inverse_transform([tempo[0]])))

temperatura_le = preprocessing.LabelEncoder()
temperatura = temperatura_le.fit_transform(df['temperatura'].values)

print("\nCategorias encontradas para temperatura: {}".format(temperatura_le.classes_))
print("Exemplo de encode de temperatura: {} e seu valor real: {}".format(temperatura[0], temperatura_le.inverse_transform([temperatura[0]])))

humidade_le = preprocessing.LabelEncoder()
humidade = humidade_le.fit_transform(df['humidade'].values)

print("\nCategorias encontradas para humidade: {}".format(humidade_le.classes_))
print("Exemplo de encode de humidade: {} e seu valor real: {}".format(humidade[0], humidade_le.inverse_transform([humidade[0]])))

vento_le = preprocessing.LabelEncoder()
vento = vento_le.fit_transform(df['vento'].values)

print("\nCategorias encontradas para vento: {}".format(vento_le.classes_))
print("Exemplo de encode de vento: {} e seu valor real: {}".format(vento[0], vento_le.inverse_transform([vento[0]])))

jogar_df = pd.DataFrame()
jogar_df['tempo'] = tempo
jogar_df['temperatura'] = temperatura
jogar_df['humidade'] = humidade
jogar_df['vento'] = vento

X = jogar_df.values

#print(X)
jogar_le = preprocessing.LabelEncoder()
y = jogar_le.fit_transform(df['jogar'].values)

#print(y)

print("Categorias encontradas para jogar: {}".format(jogar_le.classes_))
print("Exemplo de encode de jogar: {} e seu valor real: {}".format(y[0], jogar_le.inverse_transform([y[0]])))

## treinando o modelo

nb = CategoricalNB()
nb.fit(X, y)


# Com base na pergunta inicial, dado as características tempo=sol, temperatura=quente, humidade=normal e vento=sim:
print()
print("Amostra:tempo=sol, temperatura=quente, humidade=normal e vento=sim ")
nova_amostra = [
      tempo_le.transform(['sol'])[0], 
      temperatura_le.transform(['quente'])[0],
      humidade_le.transform(['normal'])[0],
      vento_le.transform(['sim'])[0]
  ]

#print("Amostra :" + nova_amostra)


saida = nb.predict([nova_amostra])
print( "E temos a classificação: " + jogar_le.inverse_transform(saida))