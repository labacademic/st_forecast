import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Forecast Holt-Winters", layout="wide")

st.title("📈 Forecast de series horizontales con Holt-Winters")

@st.cache_data
def cargar_archivo(archivo):
    if archivo.name.endswith(".csv"):
        return pd.read_csv(archivo)
    return pd.read_excel(archivo)

archivo = st.file_uploader(
    "Carga un archivo CSV o Excel",
    type=["csv", "xlsx"]
)

if archivo is None:
    st.info("Carga un archivo para iniciar.")
    st.stop()

df = cargar_archivo(archivo)

st.subheader("Vista previa")
st.dataframe(df.head())

st.sidebar.header("Configuración de la serie")

columnas = df.columns.tolist()

columnas_id = st.sidebar.multiselect(
    "Columnas descriptivas que NO son tiempo",
    columnas,
    default=columnas[:3]
)

columnas_tiempo = [col for col in columnas if col not in columnas_id]

if len(columnas_tiempo) < 3:
    st.warning("Se necesitan al menos 3 columnas de tiempo.")
    st.stop()

indice_fila = st.selectbox(
    "Selecciona la fila / serie a pronosticar",
    options=df.index.tolist(),
    format_func=lambda i: f"Fila {i} | {df.loc[i, columnas_id[0]] if columnas_id else ''}"
)

serie = df.loc[indice_fila, columnas_tiempo]
serie = pd.to_numeric(serie, errors="coerce").dropna()
serie.index = range(1, len(serie) + 1)

st.subheader("Serie seleccionada")
st.line_chart(serie)

st.sidebar.header("Modelo Holt-Winters")

trend = st.sidebar.selectbox(
    "Tendencia",
    [None, "add", "mul"],
    format_func=lambda x: "Sin tendencia" if x is None else x
)

seasonal = st.sidebar.selectbox(
    "Estacionalidad",
    [None, "add", "mul"],
    format_func=lambda x: "Sin estacionalidad" if x is None else x
)

seasonal_periods = st.sidebar.number_input(
    "Periodo estacional",
    min_value=2,
    max_value=max(2, len(serie) // 2),
    value=min(4, max(2, len(serie) // 2)),
    step=1
)

damped_trend = st.sidebar.checkbox("Tendencia amortiguada", value=False)

horizonte = st.sidebar.number_input(
    "Horizonte de pronóstico",
    min_value=1,
    max_value=52,
    value=8,
    step=1
)

if st.button("Generar forecast"):

    try:
        modelo = ExponentialSmoothing(
            serie,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods if seasonal else None,
            damped_trend=damped_trend,
            initialization_method="estimated"
        )

        ajuste = modelo.fit(optimized=True)
        forecast = ajuste.forecast(horizonte)

        forecast.index = range(len(serie) + 1, len(serie) + horizonte + 1)

        resultado = pd.DataFrame({
            "Periodo": list(serie.index) + list(forecast.index),
            "Tipo": ["Histórico"] * len(serie) + ["Forecast"] * len(forecast),
            "Valor": list(serie.values) + list(forecast.values)
        })

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(serie.index, serie.values, marker="o", label="Histórico")
            ax.plot(forecast.index, forecast.values, marker="o", linestyle="--", label="Forecast")
            ax.set_title(f"Forecast Holt-Winters - Fila {indice_fila}")
            ax.set_xlabel("Periodo")
            ax.set_ylabel("Valor")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with col2:
            st.subheader("Resultado")
            st.dataframe(resultado)

        st.download_button(
            "Descargar resultado CSV",
            data=resultado.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_fila_{indice_fila}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("No se pudo ajustar el modelo.")
        st.write(e)