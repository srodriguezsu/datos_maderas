import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Cargar el archivo GeoJSON de Colombia
colombia = gpd.read_file("https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json")

# Configuración de la página
st.set_page_config(page_title="Análisis de Datos de Madera en Colombia", layout="wide")
st.title("Análisis de Datos de Madera en Colombia")

def load_data(uploaded_file, url):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif url:
        return pd.read_csv(url)
    st.error("Por favor, sube un archivo o introduce una URL.")
    return None

def fill_missing_values(df):
    return df.interpolate(method='linear')

def most_common_species(df):
    return df['especie'].value_counts().nlargest(10)

def plot_bar_chart(data):
    fig, ax = plt.subplots()
    data.plot(kind='bar', ax=ax)
    ax.set_title('Top 10 Especies de Madera con Mayor Volumen Movilizado')
    ax.set_xlabel('Especie')
    ax.set_ylabel('Volumen Movilizado')
    st.pyplot(fig)

def plot_heatmap(df):
    required_columns = ['dpto', 'volumen_m3']
    if not all(col in df.columns for col in required_columns):
        st.error(f"El DataFrame no contiene las columnas necesarias: {required_columns}")
        return
    df_volume = df.groupby('dpto', as_index=False)['volumen_m3'].sum()
    df_volume['dpto'] = df_volume['dpto'].str.upper()
    colombia_volume = colombia.merge(df_volume, how='left', left_on='NOMBRE_DPT', right_on='dpto')
    fig, ax = plt.subplots(figsize=(10, 8))
    colombia_volume.plot(column='volumen_m3', cmap='OrRd', legend=True, ax=ax, missing_kwds={"color": "lightgray", "label": "Sin datos"})
    ax.set_title('Distribución de Volúmenes de Madera por Departamento')
    ax.set_axis_off()
    st.pyplot(fig)

def plot_top_municipalities(df):
    required_columns = ['municipio', 'volumen_m3']
    if not all(col in df.columns for col in required_columns):
        st.error(f"El DataFrame no contiene las columnas necesarias: {required_columns}")
        return
    df_volume = df.groupby('municipio', as_index=False)['volumen_m3'].sum().nlargest(10)
    df_volume['municipio'] = df_volume['municipio'].str.upper()
    municipios = gpd.read_file("https://raw.githubusercontent.com/macortesgu/MGN_2021_geojson/refs/heads/main/MGN2021_MPIO_web.geo.json")
    municipios['MPIO_CNMBR'] = municipios['MPIO_CNMBR'].str.upper()
    municipios_volume = municipios.merge(df_volume, how='left', left_on='MPIO_CNMBR', right_on='municipio')
    fig, ax = plt.subplots(figsize=(10, 8))
    municipios_volume.plot(column='volumen_m3', cmap='Reds', legend=True, ax=ax, missing_kwds={"color": "lightgray", "label": "Sin datos"})
    ax.set_title('Top 10 Municipios con Mayor Movilización de Madera')
    ax.set_axis_off()
    st.pyplot(fig)

def plot_temporal_evolution(df):
    temporal_evolution = df.groupby([df['semestre'], 'especie'])['volumen_m3'].sum().unstack()
    fig, ax = plt.subplots()
    temporal_evolution.plot(ax=ax)
    ax.set_title('Evolución Temporal del Volumen de Madera por Especie')
    ax.set_xlabel('Año')
    ax.set_ylabel('Volumen Movilizado')
    st.pyplot(fig)

def identify_outliers(df):
    z_scores = np.abs(stats.zscore(df['volumen_m3']))
    return df[z_scores > 3]

def group_by_municipality(df):
    return df.groupby('municipio', as_index=False)['volumen_m3'].sum()

def least_common_species_map(df):
    least_common_species = df['especie'].value_counts().nsmallest(10).index
    df_filtered = df[df['especie'].isin(least_common_species)]
    df_filtered['municipio'] = df_filtered['municipio'].str.upper()
    df_pivot = pd.pivot_table(df_filtered, values='volumen_m3', index='municipio', columns='especie', aggfunc='count', fill_value=0)
    municipios = gpd.read_file("https://raw.githubusercontent.com/macortesgu/MGN_2021_geojson/refs/heads/main/MGN2021_MPIO_web.geo.json")
    municipios['MPIO_CNMBR'] = municipios['MPIO_CNMBR'].str.upper()
    municipios_species = municipios.merge(df_pivot, how='left', left_on='MPIO_CNMBR', right_index=True)
    municipios_species['total_especies_menos_comunes'] = municipios_species[least_common_species].sum(axis=1)
    selected_species = st.selectbox("Selecciona una especie para visualizar:", options=least_common_species)
    fig, ax = plt.subplots(figsize=(12, 8))
    municipios_species.plot(column=selected_species, cmap='viridis', legend=True, ax=ax, missing_kwds={"color": "lightgray", "label": "Sin datos"})
    ax.set_title(f'Distribución de {selected_species}')
    ax.set_axis_off()
    st.pyplot(fig)

def apply_clustering(df):
    required_columns = ['municipio', 'volumen_m3']
    if not all(col in df.columns for col in required_columns):
        st.error(f"El DataFrame no contiene las columnas necesarias: {required_columns}")
        return
    df_volume = df.groupby('municipio', as_index=False)['volumen_m3'].sum()
    df_volume['municipio'] = df_volume['municipio'].str.upper()
    municipios = gpd.read_file("https://raw.githubusercontent.com/macortesgu/MGN_2021_geojson/refs/heads/main/MGN2021_MPIO_web.geo.json")
    municipios['MPIO_CNMBR'] = municipios['MPIO_CNMBR'].str.upper()
    municipios_volume = municipios.merge(df_volume, how='left', left_on='MPIO_CNMBR', right_on='municipio')
    municipios_volume['volumen_m3'].fillna(0, inplace=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(municipios_volume[['volumen_m3']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    municipios_volume['cluster'] = kmeans.fit_predict(scaled_data)
    fig, ax = plt.subplots(figsize=(10, 8))
    municipios_volume.plot(column='cluster', cmap='viridis', legend=True, ax=ax, missing_kwds={"color": "lightgray", "label": "Sin datos"})
    ax.set_title('Clustering de Municipios por Volumen de Madera')
    ax.set_axis_off()
    st.pyplot(fig)

def shannon_diversity_index(df):
    species_counts = df['especie'].value_counts()
    proportions = species_counts / species_counts.sum()
    return -np.sum(proportions * np.log(proportions))

def main():
    st.sidebar.title("Opciones de Carga de Datos")
    uploaded_file = st.sidebar.file_uploader("Sube un archivo CSV", type=["csv"])
    url = st.sidebar.text_input("O introduce una URL de un archivo CSV")
    df = load_data(uploaded_file, url)
    if df is not None:
        df = fill_missing_values(df)
        st.write("Datos cargados y valores faltantes rellenados:")
        st.write(df)
        st.header("Especies de Madera Más Comunes")
        common_species = most_common_species(df)
        st.write(common_species)
        plot_bar_chart(common_species)
        st.header("Mapa de Calor de Volúmenes por Departamento")
        plot_heatmap(df)
        st.header("Top 10 Municipios con Mayor Movilización de Madera")
        plot_top_municipalities(df)
        st.header("Evolución Temporal del Volumen de Madera por Especie")
        plot_temporal_evolution(df)
        st.header("Identificación de Outliers")
        outliers = identify_outliers(df)
        st.write(outliers)
        st.header("Especies de Madera con Menor Volumen Movilizado")
        least_common_species_map(df)
        st.header("Clustering de Departamentos/Municipios")
        apply_clustering(df)
        st.header("Índice de Diversidad de Shannon por Departamento")
        diversity_index = shannon_diversity_index(df)
        st.write(f"Índice de Diversidad de Shannon: {diversity_index}")

if __name__ == "__main__":
    main()
