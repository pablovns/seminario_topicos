import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Regressão linear


trad_dict = {
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
st.header("Dados sobre aluguel de imóveis em diferentes cidades do Brasil")
df = pd.read_csv("houses_to_rent_v2.csv")
df.rename(columns=trad_dict, inplace=True) # traduz os nomes das colunas para melhor visualização
df

# cria um df com apenas as colunas com dados quantitativos 
df_quant = df.drop(['Cidade', 'Permite animais', 'Mobiliado', 'Taxa de condomínio (R$)', 'Valor do aluguel (R$)', 'IPTU (R$)', 'Seguro contra incêndio (R$)', 'Total (R$)'], axis=1)

# apresentar estatísticas da base de dados
# plota a quantidade de casas em cada uma das 5 cidades
fig, ax = plt.subplots()
df['Cidade'].value_counts().sort_values().plot(ax=ax, kind='barh')
ax.set_ylabel('Cidade')
ax.set_xlabel('Casas para alugar')
st.pyplot(fig)

st.header("Estatísticas do aluguel em cada cidade")
col_quant = ['Área (m²)', 'Nº de quartos', 'Nº de banheiros', 'Vagas de estacionamento', 'Andar', 'Valor do aluguel (R$)', 'IPTU (R$)', 'Taxa de condomínio (R$)', 'Seguro contra incêndio (R$)']

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16,16)) # cria uma figura com 3 linhas e 2 colunas (6 gráficos no total)
for i, row in enumerate(np.array(col_quant[:6]).reshape(3, 2)): # redimensiona o array para as mesmas dimensões da figura
    for j, elem in enumerate(row): # percorre cada coluna da linha atualmente sendo percorrida pelo for acima
        media_col = df.groupby('Cidade')[elem].mean() # calcula e agrupa a média de cada cidade
        media_col.sort_index().plot.barh(ax=ax[i][j]) # plota a média calculada no eixo ax na linha i e na coluna j
        ax[i][j].set_title(elem)
st.pyplot(fig)

# verificar as correlações entre as variáveis
st.header("Correlação entre as variáveis quantitativas e o valor do aluguel")
for col in df_quant.columns:
    corr = df_quant[col].corr(df['Valor do aluguel (R$)']) # calcula a correlação entre a coluna e o valor do aluguel
    st.markdown(f"Correlação de **{col}** com o valor do aluguel: **{corr:.4f}**")

# escolha dos parâmetros de entrada para a predição
st.header("Predição utilizando Regressão Linear Múltipla")
opcoes = list(df_quant.columns)
container = st.container()
if st.checkbox('Selecionar todos'):
    selecao = container.multiselect('Escolha o(s) parâmetro(s) de entrada para a predição: ', options=opcoes, default=opcoes)
else:
    selecao = container.multiselect('Escolha o(s) parâmetro(s) de entrada para a predição: ', options=opcoes)

# regressão linear múltipla
rl = LinearRegression()

if selecao:
    indep = df[selecao].values.reshape(-1, len(selecao)) # variável independente
    dep = df['Valor do aluguel (R$)'].values.flatten() # variável dependente
    rl.fit(indep, dep)

    st.subheader('Insira os dados abaixo')
    x = [st.number_input(elem, min_value=0) for elem in selecao] # obtém do usuário os valores de entrada para a predição 

    x_arr = np.array(x).reshape(-1, len(selecao)) # converte para o formato adequado
    y_pred = rl.predict(x_arr) # calcula a predição
    st.markdown(f"#### Preço estimado do aluguel: R$ {float(y_pred):.2f}")
