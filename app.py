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
import geopandas as gpd

# Cargar el archivo GeoJSON de Colombia
colombia = gpd.read_file("https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/3aadedf47badbdac823b00dbe259f6bc6d9e1899/colombia.geo.json")

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

def plot_heatmap(df):
    try:
        # Verificar que las columnas necesarias estén presentes
        required_columns = ['dpto', 'volumen_m3']
        if not all(col in df.columns for col in required_columns):
            st.error(f"El DataFrame no contiene las columnas necesarias: {required_columns}")
            return

        # Agrupar los datos por departamento y sumar el volumen
        df_volume = df.groupby('dpto')['volumen_m3'].sum().reset_index()

        # Unir los datos de volumen con el GeoDataFrame de Colombia
        colombia_volume = colombia.merge(df_volume, how='left', left_on='NOMBRE_DPT', right_on='dpto')

        # Crear el mapa de calor
        fig, ax = plt.subplots(figsize=(10, 8))
        colombia_volume.plot(column='volumen_m3', cmap='OrRd', legend=True, ax=ax,
                             missing_kwds={"color": "lightgray", "label": "Sin datos"})
        ax.set_title('Distribución de Volúmenes de Madera por Departamento')
        ax.set_axis_off()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al generar el mapa de calor: {e}")


def plot_top_municipalities(df):
    try:
        # Verificar que las columnas necesarias estén presentes
        required_columns = ['municipio', 'volumen_m3']
        if not all(col in df.columns for col in required_columns):
            st.error(f"El DataFrame no contiene las columnas necesarias: {required_columns}")
            return

        # Agrupar los datos por municipio y sumar el volumen
        df_volume = df.groupby('municipio')['volumen_m3'].sum().nlargest(10).reset_index()

        # Unir los datos de volumen con el GeoDataFrame de municipios
        municipios = gpd.read_file("https://raw.githubusercontent.com/macortesgu/MGN_2021_geojson/refs/heads/main/MGN2021_MPIO_web.geo.json")  # Asegúrate de tener este archivo
        municipios_volume = municipios.merge(df_volume, how='left', left_on='MPIO_CNMBR', right_on='municipio')

        # Crear el mapa
        fig, ax = plt.subplots(figsize=(10, 8))
        municipios_volume.plot(ax=ax, color='lightgray')  # Fondo de todos los municipios
        municipios_volume.dropna().plot(column='volumen_m3', cmap='viridis', legend=True, ax=ax)  # Municipios top
        ax.set_title('Top 10 Municipios con Mayor Movilización de Madera')
        ax.set_axis_off()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al generar el mapa de municipios: {e}")

# Función para analizar la evolución temporal del volumen de madera
def plot_temporal_evolution(df):
    temporal_evolution = df.groupby([df['semestre'], 'especie'])['volumen_m3'].sum().unstack()
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
    species_distribution = df.groupby(['dpto', 'especie'])['volumen_m3'].sum().unstack()
    fig, ax = plt.subplots(figsize=(12, 8))
    species_distribution.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Distribución de Especies de Madera por Departamento')
    ax.set_xlabel('Departamento')
    ax.set_ylabel('Volumen Movilizado')
    st.pyplot(fig)

def apply_clustering(df):
    try:
        # Verificar que las columnas necesarias estén presentes
        required_columns = ['municipio', 'volumen_m3']
        if not all(col in df.columns for col in required_columns):
            st.error(f"El DataFrame no contiene las columnas necesarias: {required_columns}")
            return

        # Agrupar los datos por municipio y sumar el volumen
        df_volume = df.groupby('municipio')['volumen_m3'].sum().reset_index()

        # Cargar el archivo GeoJSON de municipios de Colombia
        url_municipios = "https://raw.githubusercontent.com/macortesgu/MGN_2021_geojson/refs/heads/main/MGN2021_MPIO_web.geo.json"
        municipios = gpd.read_file(url_municipios)

        # Unir los datos de volumen con el GeoDataFrame de municipios
        municipios_volume = municipios.merge(df_volume, how='left', left_on='MPIO_CNMBR', right_on='municipio')

        # Verificar si hay datos después de la unión
        if municipios_volume.empty:
            st.error("No hay datos válidos después de la unión. Verifica los nombres de los municipios.")
            return

        df['volumen_m3'] = pd.to_numeric(df['volumen_m3'], errors='coerce')  
        # Rellenar valores faltantes en 'volumen_m3' con 0
        municipios_volume['volumen_m3'] = municipios_volume['volumen_m3'].fillna(0)

        # Verificar que haya datos para escalar
        if municipios_volume['volumen_m3'].sum() == 0:
            st.error("No hay datos de volumen para aplicar clustering.")
            return

        # Escalar los datos para el clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(municipios_volume[['volumen_m3']])

        # Aplicar KMeans clustering
        kmeans = KMeans(n_clusters=3)  # Puedes ajustar el número de clusters
        municipios_volume['cluster'] = kmeans.fit_predict(scaled_data)

        # Crear el mapa de clusters
        fig, ax = plt.subplots(figsize=(10, 8))
        municipios_volume.plot(column='cluster', cmap='viridis', legend=True, ax=ax,
                               missing_kwds={"color": "lightgray", "label": "Sin datos"})
        ax.set_title('Clustering de Municipios por Volumen de Madera')
        ax.set_axis_off()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error al aplicar clustering: {e}")

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
