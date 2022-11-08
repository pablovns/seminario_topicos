import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree


trad = {
    'city': 'Cidade',
    'area': 'Área (m²)',
    'rooms': 'Nº de quartos',
    'bathroom': 'Nº de banheiros',
    'parking spaces': 'Vagas de estacionamento',
    'floor': 'Andar',
    'animal': 'Permite animais',
    'furniture': 'Mobiliado',
    'hoa (R$)': 'Taxa de condomínio (R$)',
    'rent amount (R$)': 'Valor do aluguel (R$)',
    'property tax (R$)': 'IPTU (R$)',
    'fire insurance (R$)': 'Seguro contra incêndio (R$)',
    'total (R$)': 'Total (R$)'
}

st.title('Seminário de Tópicos Especiais em Informática') 
df = pd.read_csv("houses_to_rent_v2.csv")
df.rename(columns=trad, inplace=True) # traduz os nomes das colunas para melhor visualização
st.dataframe(df.head(3))
df.drop(['Taxa de condomínio (R$)', 'Total (R$)', 'IPTU (R$)', 'Seguro contra incêndio (R$)'], axis=1, inplace=True)
aux = st.multiselect('Escolha o (s) parâmetro (s) de entrada para a predição', options=df.drop(['Valor do aluguel (R$)'], axis=1).columns)

# Variáveis
alvo = 'Valor do aluguel (R$)' #variável alvo
quant = [] #Variáveis quantitativas
quali = [] #Variáveis qualitativas
vals = {} # Valores para a predição

base = df[aux].join(df['Valor do aluguel (R$)'])

st.subheader('Insira os dados abaixo')
for elem in aux:
    if df[elem].dtype != 'object': # checa se a coluna é de dados númericos (quantitativos)
        quant.append(elem)
        vals[elem] = st.number_input(elem, min_value=0)
    else:
        quali.append(elem)
        vals[elem] = st.radio(elem, options=df[elem].unique())

# Árvore
arv = tree.DecisionTreeClassifier() #árvore de decisão
arv.fit(pd.concat([pd.get_dummies(base[quali]), base[quant]], axis=1), base[alvo]) #cria a árvore

res = arv.predict([vals['Valor do aluguel (R$)']])
res