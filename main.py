import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import geobr


st.title('Dados do clima brasileiro') #Título para a página
df_south = pd.read_csv("north.csv")
st.dataframe(df_south.head(10))
