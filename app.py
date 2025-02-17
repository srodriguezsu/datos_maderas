import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Análisis de Datos de Madera en Colombia", layout="wide")

# Título de la aplicación
st.title("Análisis de Datos de Madera en Colombia")

# Función para cargar datos
def load_data(uploaded_file, url):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif url:
        df = pd.read_csv(url)
    else:
        st.error("Por favor, sube un archivo o introduce una URL.")
        return None
    return df

# Función para rellenar valores faltantes mediante interpolación
def fill_missing_values(df):
    df = df.interpolate(method='linear')
    return df

# Función para identificar las especies de madera más comunes
def most_common_species(df):
    common_species = df['especie'].value_counts().nlargest(10)
    return common_species

# Función para crear un gráfico de barras
def plot_bar_chart(data):
    fig, ax = plt.subplots()
    data.plot(kind='bar', ax=ax)
    ax.set_title('Top 10 Especies de Madera con Mayor Volumen Movilizado')
    ax.set_xlabel('Especie')
    ax.set_ylabel('Volumen Movilizado')
    st.pyplot(fig)

# Función para generar un mapa de calor
def plot_heatmap(df):
    heatmap_data = df.pivot_table(index='dpto', columns='especie', values='volumen_m3', aggfunc='sum')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', ax=ax)
    ax.set_title('Distribución de Volúmenes de Madera por Departamento')
    st.pyplot(fig)

# Función para visualizar en un mapa los municipios con mayor movilización de madera
def plot_top_municipalities(df):
    try:
        # Verifica que las columnas necesarias estén presentes
        required_columns = ['municipio', 'volumen_m3']
        if not all(col in df.columns for col in required_columns):
            st.error(f"El DataFrame no contiene las columnas necesarias: {required_columns}")
            return

        # Agrupa por municipio y suma el volumen movilizado
        top_municipalities = df.groupby('municipio')['volumen_m3'].sum().nlargest(10).reset_index()

        # Si no hay coordenadas, muestra un mensaje de error
        if 'latitud' not in df.columns or 'longitud' not in df.columns:
            st.error("El DataFrame no contiene las columnas 'latitud' y 'longitud'. No se puede generar el mapa.")
            return

        # Crea el mapa
        fig = px.scatter_geo(
            top_municipalities,
            lat='latitud',
            lon='longitud',
            size='volumen_m3',
            hover_name='municipio',
            scope='south america',
            title='Top 10 Municipios con Mayor Movilización de Madera'
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error al generar el mapa de municipios: {e}")

# Función para analizar la evolución temporal del volumen de madera
def plot_temporal_evolution(df):
    df['fecha'] = pd.to_datetime(df['fecha'])
    temporal_evolution = df.groupby([df['fecha'].dt.year, 'especie'])['volumen_m3'].sum().unstack()
    fig, ax = plt.subplots()
    temporal_evolution.plot(ax=ax)
    ax.set_title('Evolución Temporal del Volumen de Madera por Especie')
    ax.set_xlabel('Año')
    ax.set_ylabel('Volumen Movilizado')
    st.pyplot(fig)

# Función para identificar outliers
def identify_outliers(df):
    z_scores = np.abs(stats.zscore(df['volumen_m3']))
    outliers = df[z_scores > 3]
    return outliers

# Función para agrupar datos por municipio y calcular el volumen total
def group_by_municipality(df):
    municipality_volume = df.groupby('municipio')['volumen_m3'].sum().reset_index()
    return municipality_volume

# Función para identificar especies con menor volumen movilizado
def least_common_species(df):
    least_common = df['especie'].value_counts().nsmallest(10)
    return least_common

# Función para comparar la distribución de especies entre departamentos
def compare_species_distribution(df):
    species_distribution = df.groupby(['departamento', 'especie'])['volumen_m3'].sum().unstack()
    fig, ax = plt.subplots(figsize=(12, 8))
    species_distribution.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Distribución de Especies de Madera por Departamento')
    ax.set_xlabel('Departamento')
    ax.set_ylabel('Volumen Movilizado')
    st.pyplot(fig)

# Función para aplicar clustering y mostrar clusters en un mapa
def apply_clustering(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['latitud', 'longitud', 'volumen_m3']])
    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    fig = px.scatter_geo(df, lat='latitud', lon='longitud', color='cluster', scope='south america')
    st.plotly_chart(fig)

# Función para calcular el índice de diversidad de Shannon
def shannon_diversity_index(df):
    species_counts = df['especie'].value_counts()
    proportions = species_counts / species_counts.sum()
    diversity_index = -np.sum(proportions * np.log(proportions))
    return diversity_index

# Función principal
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

        st.header("Volumen Total de Madera por Municipio")
        municipality_volume = group_by_municipality(df)
        st.write(municipality_volume)

        st.header("Especies de Madera con Menor Volumen Movilizado")
        least_common = least_common_species(df)
        st.write(least_common)

        st.header("Comparación de Distribución de Especies entre Departamentos")
        compare_species_distribution(df)

        st.header("Clustering de Departamentos/Municipios")
        apply_clustering(df)

        st.header("Índice de Diversidad de Shannon por Departamento")
        diversity_index = shannon_diversity_index(df)
        st.write(f"Índice de Diversidad de Shannon: {diversity_index}")

if __name__ == "__main__":
    main()
