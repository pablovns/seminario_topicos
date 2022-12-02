import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Regressão linear


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
st.dataframe(df)
st.write(df['Cidade'].value_counts())

ALVO = df['Valor do aluguel (R$)']
colunas_drop = ['Cidade', 'Permite animais', 'Mobiliado', 'Taxa de condomínio (R$)', 'Valor do aluguel (R$)', 'IPTU (R$)', 'Seguro contra incêndio (R$)', 'Total (R$)']
df_clean = df.drop(colunas_drop, axis=1)

# verificar as correlações entre as variáveis
for col in df_clean.columns:
    st.markdown(f"Correlação de **{col}** com o valor do aluguel: **{df_clean[col].corr(ALVO):.4f}**")

st.header("Predição utilizando Regressão Linear Múltipla")
opcoes = list(df_clean.columns)
container = st.container()
todos = st.checkbox('Selecionar todos')
str_input = 'Escolha o(s) parâmetro(s) de entrada para a predição: '
if todos:
    selecao = container.multiselect(str_input, options=opcoes, default=opcoes)
    # caso o usuário marque a opção, cria um multiselect com as opções já selecionadas (parâmetro 'default')
else:
    selecao = container.multiselect(str_input, options=opcoes)
    # caso contrário, cria um multiselect sem nada selecionado
  
# regressão linear múltipla
rl = LinearRegression()
indep = df[selecao].values.reshape(-1, len(selecao)) # variável independente
dep = ALVO.values.flatten() # variável dependente
rl.fit(indep, dep)

if selecao:
    st.subheader('Insira os dados abaixo')
    x = [st.number_input(elem, min_value=0) for elem in selecao] # obtém os valores para a variável independente 

    x_arr = np.array(x).reshape(-1, len(selecao)) # converte para o formato adequado
    y_pred = rl.predict(x_arr) # calcula a predição
    st.markdown(f"#### Preço estimado do aluguel: R$ {float(y_pred):.2f}")
