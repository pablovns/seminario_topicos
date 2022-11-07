import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression #Regressão linear


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
opcoes = df.drop(['Permite animais', 'Mobiliado', 'Cidade'], axis=1).columns
aux = st.multiselect('Escolha o (s) parâmetro (s) de entrada para a predição', options=opcoes)

# predição
X = df[aux].values # variável independente
Y = df['Valor do aluguel (R$)'].values # variavel dependente

rl = LinearRegression()
rl.fit(X, Y)

x = [] # valor para a variável independente
st.subheader('Insira os dados abaixo')
for elem in aux:
    num = st.number_input(f'{elem}', min_value=0)
    x.append([num])

x_arr = np.array(x).reshape(-1, len(aux))
y_pred = rl.predict(x_arr)
st.text(f"O valor estimado do aluguel é de: {float(y_pred[0])}")