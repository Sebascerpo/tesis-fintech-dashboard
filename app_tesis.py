import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.stats import kruskal, spearmanr
import os # <--- Agrega esto al inicio junto a los otros imports
# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Análisis Fintech Colombia", layout="wide")
template_style = "simple_white"


# --- MÓDULO DE PROCESAMIENTO DE DATOS (ACTUALIZADO) ---
@st.cache_data
def get_data(file_path):
    # 1. Carga agnóstica al encoding
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    df.dropna(how="all", inplace=True)

    # 2. Diccionario de variables (EXPANDIDO)
    renames = {
        # Variables Objetivo y Predictoras (Ya existentes)
        "11. En general, ¿qué tan probable es que en los próximos 12 meses continúes usando o recomiendes a otras personas las billeteras o créditos digitales?": "y_intencion",
        "5. En general, ¿qué tan seguro(a) te sientes al usar servicios Fintech para manejar tu dinero o tus datos personales?": "x_seguridad",
        "12. ¿Cómo calificas tu nivel de conocimiento o educación financiera para manejar servicios Fintech?": "x_educacion_fin",
        "8.  ¿Cuáles de los siguientes factores influyen más en tu confianza o desconfianza hacia las plataformas Fintech?": "txt_factores",
        # Demográficos
        "13. Selecciona tu rango de edad:": "d_edad",
        "15. Selecciona tu género": "d_genero",
        # Herramientas (NUEVO)
        " 1. En relación con el servicio Nequi, selecciona la opción que te describa mejor:": "uso_nequi",
        "2. En relación con el servicio Daviplata, selecciona la opción que te describa mejor:": "uso_daviplata",
        " 3. En relación con el servicio Addi, selecciona la opción que te describa mejor:": "uso_addi",
        "4. En relación con el servicio Sistecrédito, selecciona la opción que te describa mejor:": "uso_sistecredito",
        # Experiencia y Comparación (NUEVO)
        "9. ¿Has tenido alguna experiencia negativa al usar billeteras o créditos digitales (errores, fraudes, demoras, mala atención)?": "exp_negativa",
        "10. En tu opinión, ¿las plataformas Fintech ofrecen una protección de datos personales igual, mejor o peor que la banca tradicional?": "comp_banca",
    }

    # Normalización de nombres
    df.columns = df.columns.str.strip()
    df.rename(columns={k.strip(): v for k, v in renames.items()}, inplace=True)

    # 3. Mapeos Numéricos (Igual que antes)
    map_seguridad = {
        "Nada seguro(a).": 1,
        "Poco seguro(a)": 2,
        "Poco seguro(a).": 2,
        "Neutral": 3,
        "Neutral.": 3,
        "Seguro(a).": 4,
        "Muy seguro(a).": 5,
    }

    map_intencion = {
        "Nada probable.": 1,
        "Poco probable.": 2,
        "Indiferente.": 3,
        "Neutral": 3,
        "Neutral.": 3,
        "Probable.": 4,
        "Muy probable.": 5,
    }

    map_edu = {"Nulo": 1, "Bajo": 2, "Medio": 3, "Alto": 4, "Muy alto": 5}

    # Aplicar mapeos principales
    df["num_seguridad"] = df["x_seguridad"].map(map_seguridad)
    df["num_intencion"] = df["y_intencion"].map(map_intencion)
    df["num_educacion"] = df["x_educacion_fin"].map(map_edu)

    # 4. Procesamiento de Texto (Factores)
    if "txt_factores" in df.columns:
        factors_list = df["txt_factores"].dropna().str.split(",").tolist()
        flat_list = [
            item.strip().rstrip(".") for sublist in factors_list for item in sublist
        ]
        unique_factors = list(set(flat_list))

        for factor in unique_factors:
            if len(factor) > 5:
                df[f"factor_{factor}"] = df["txt_factores"].apply(
                    lambda x: 1 if isinstance(x, str) and factor in x else 0
                )

    return df


# --- INTERFAZ PRINCIPAL ---
st.title("Análisis de Determinantes en la Adopción Fintech")
st.markdown(
    """
**Tablero de Control Profesional - Trabajo de Grado**
Este sistema permite explorar interactivamente las relaciones estadísticas entre la percepción de seguridad, 
el perfil demográfico y la decisión de adopción de tecnologías financieras en Colombia.
"""
)

ARCHIVO_POR_DEFECTO = 'datos.csv'

df = None

# 1. Intentar cargar archivo local automáticamente
if os.path.exists(ARCHIVO_POR_DEFECTO):
    try:
        df = get_data(ARCHIVO_POR_DEFECTO)
        st.success(f"Datos cargados automáticamente desde: {ARCHIVO_POR_DEFECTO}")
    except Exception as e:
        st.error(f"Error al cargar el archivo local: {e}")

# 2. Si no se encontró el automático, o si el usuario quiere cambiarlo
if df is None:
    st.warning(f"No se encontró el archivo '{ARCHIVO_POR_DEFECTO}'. Por favor cárgalo manualmente.")
    uploaded_file = st.sidebar.file_uploader("Cargar Dataset (CSV)", type=['csv'])
    if uploaded_file:
        df = get_data(uploaded_file)

# --- INICIO DEL DASHBOARD ---
if df is not None:
    # ... AQUÍ SIGUE EL RESTO DEL CÓDIGO (Tabs, Gráficos, etc.)
    # A partir de aquí copias todo lo que tenías dentro del "if uploaded_file:" anterior

    # Definición de Tabs
    tab1, tab2, tab3 = st.tabs(
        ["Análisis Descriptivo", "Inferencia Estadística", "Modelado Predictivo"]
    )

    # --- TAB 1: DESCRIPTIVO (EXPANDIDO) ---
# --- TAB 1: ANÁLISIS DESCRIPTIVO (DISEÑO MEJORADO) ---
    with tab1:
        st.header("Panorama General de la Muestra y Adopción")

        # BLOQUE 1: DEMOGRAFÍA
        st.subheader("1. Caracterización Demográfica")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            # Gráfico de Edades
            df_edad = df["d_edad"].value_counts().reset_index()
            df_edad.columns = ["Rango de Edad", "Frecuencia"]
            
            fig_edad = px.bar(
                df_edad,
                x="Rango de Edad",
                y="Frecuencia",
                title="Distribución por Rango de Edad",
                # CAMBIO CLAVE: Coloreamos por la etiqueta (Edad) y no por el número
                color="Rango de Edad", 
                # "Prism": Una paleta de alto contraste (colores muy distintos entre sí)
                color_discrete_sequence=px.colors.qualitative.Prism,
                template=template_style,
            )
            
            # Opcional: Ocultar la leyenda si ya se ve en el eje X para limpiar visualmente
            fig_edad.update_layout(showlegend=False)
            st.plotly_chart(fig_edad, use_container_width=True)

        with col_d2:
            # Gráfico de Género
            fig_gen = px.pie(
                df,
                names="d_genero",
                title="Composición por Género",
                # "Set3": Colores pastel distintivos pero suaves a la vista
                color_discrete_sequence=px.colors.qualitative.Set3,
                template=template_style,
                hole=0.4,
            )
            st.plotly_chart(fig_gen, use_container_width=True)

        st.markdown("---")

        # BLOQUE 2: PENETRACIÓN DE MERCADO
        st.subheader("2. Penetración de Herramientas Fintech")
        st.markdown(
            "Comparativa de usuarios activos ('Lo uso actualmente') entre billeteras y créditos."
        )

        # Cálculo de métricas de adopción
        tools = ["uso_nequi", "uso_daviplata", "uso_addi", "uso_sistecredito"]
        tool_names = ["Nequi", "Daviplata", "Addi", "Sistecrédito"]
        adoption_rates = []

        for tool in tools:
            if tool in df.columns:
                count = df[
                    df[tool]
                    .astype(str)
                    .str.contains("actualmente", case=False, na=False)
                ].shape[0]
                adoption_rates.append(count)
            else:
                adoption_rates.append(0)

        df_adoption = pd.DataFrame(
            {"Herramienta": tool_names, "Usuarios Activos": adoption_rates}
        )
        df_adoption = df_adoption.sort_values("Usuarios Activos", ascending=False)

        fig_adopt = px.bar(
            df_adoption,
            x="Herramienta",
            y="Usuarios Activos",
            text="Usuarios Activos",
            color="Herramienta",
            title="Cuota de Mercado en la Muestra (Usuarios Activos)",
            # "Bold": Colores fuertes y sólidos para distinguir marcas claramente
            color_discrete_sequence=px.colors.qualitative.Bold,
            template=template_style,
        )
        st.plotly_chart(fig_adopt, use_container_width=True)

        st.markdown("---")

        # BLOQUE 3: SEGURIDAD Y EXPERIENCIA
        st.subheader("3. Percepción de Seguridad y Experiencia")
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            st.markdown("**Distribución de Seguridad**")
            fig_sec = px.histogram(
                df,
                x="num_seguridad",
                nbins=5,
                title="Percepción General (1-5)",
                # Un color Índigo sólido (#4F46E5) que denota tecnología/seriedad
                color_discrete_sequence=["#4F46E5"],
                template=template_style,
            )
            fig_sec.update_layout(
                xaxis_title="Nivel de Seguridad", yaxis_title="Frecuencia"
            )
            st.plotly_chart(fig_sec, use_container_width=True)

        with col_s2:
            st.markdown("**Fintech vs. Banca Tradicional**")
            if "comp_banca" in df.columns:
                df_comp = df["comp_banca"].value_counts().reset_index()
                df_comp.columns = ["Opinión", "Conteo"]
                fig_comp = px.pie(
                    df_comp,
                    names="Opinión",
                    values="Conteo",
                    title="Protección de Datos: Comparativa",
                    # "Safe": Paleta diseñada para ser clara y agradable
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

        with col_s3:
            st.markdown("**Incidencia de Fraudes/Errores**")
            if "exp_negativa" in df.columns:
                df_exp = df["exp_negativa"].value_counts().reset_index()
                df_exp.columns = ["Experiencia Negativa", "Conteo"]
                
                fig_exp = px.bar(
                    df_exp,
                    x="Experiencia Negativa",
                    y="Conteo",
                    title="Reporte de Incidentes",
                    color="Conteo",
                    # "YlOrRd": De Amarillo a Rojo Intenso (Alerta)
                    color_continuous_scale="YlOrRd",
                )
                fig_exp.update_xaxes(showticklabels=False)
                st.plotly_chart(fig_exp, use_container_width=True)

        # BLOQUE 4: FACTORES
        st.markdown("---")
        st.markdown("**Factores Determinantes de Confianza**")
        factor_cols = [c for c in df.columns if c.startswith("factor_")]
        factor_counts = df[factor_cols].sum().sort_values(ascending=True)
        factor_counts.index = factor_counts.index.str.replace("factor_", "")

        fig_factors = px.bar(
            x=factor_counts.values,
            y=factor_counts.index,
            orientation="h",
            color=factor_counts.values,
            # "Viridis": La paleta más profesional en ciencia de datos (Azul a Amarillo)
            color_continuous_scale="Viridis",
            title="Desglose de Factores de Confianza (Múltiple Respuesta)",
            template=template_style,
        )
        st.plotly_chart(fig_factors, use_container_width=True)

    # --- TAB 2: INFERENCIA ---
    with tab2:
        st.subheader("Pruebas de Hipótesis")

        # 1. Correlación Robusta
        st.markdown("#### 1. ¿Existe correlación entre Seguridad e Intención de Uso?")
        st.write(
            "Se utiliza el coeficiente de **Spearman** dado que las variables son ordinales (no siguen una distribución normal perfecta)."
        )

        df_corr = df[["num_seguridad", "num_intencion", "num_educacion"]].dropna()
        corr, p_value = spearmanr(df_corr["num_seguridad"], df_corr["num_intencion"])

        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Coeficiente Rho de Spearman", f"{corr:.4f}")
        col_res2.metric("Valor P (Significancia)", f"{p_value:.4e}")

        if p_value < 0.05:
            st.info(
                f"**Interpretación:** Existe una relación estadísticamente significativa (p < 0.05). La fuerza de la relación es de {corr:.2f}, lo cual indica una correlación positiva moderada."
            )
        else:
            st.error(
                "Interpretación: No existe evidencia estadística suficiente para afirmar una relación."
            )

        st.markdown("---")

        # 2. Diferencia de Grupos (Kruskal-Wallis)
        st.markdown("#### 2. ¿Varía la percepción de seguridad según el Género?")
        st.write(
            "Se aplica la prueba **U de Mann-Whitney** (o Kruskal-Wallis) para comparar las medianas de seguridad entre géneros."
        )

        generos = df["d_genero"].unique()
        groups = [
            df[df["d_genero"] == g]["num_seguridad"].dropna()
            for g in generos
            if isinstance(g, str)
        ]

        if len(groups) > 1:
            stat_k, p_k = kruskal(*groups)
            st.write(f"**Estadístico H:** {stat_k:.4f}, **Valor P:** {p_k:.4f}")
            if p_k > 0.05:
                st.write(
                    "Resultados: **No hay diferencias significativas** en la percepción de seguridad basadas en el género. El miedo o confianza es transversal."
                )
            else:
                st.write(
                    "Resultados: **Existen diferencias significativas** entre géneros."
                )

            # Boxplot para visualizar
            fig_box = px.box(
                df,
                x="d_genero",
                y="num_seguridad",
                color="d_genero",
                template=template_style,
                title="Distribución de Seguridad por Género",
            )
            st.plotly_chart(fig_box, use_container_width=True)

    # --- TAB 3: MODELADO AVANZADO ---
# --- TAB 3: MODELADO PREDICTIVO (VERSIÓN EXPLICATIVA Y VISUAL) ---
    with tab3:
        st.header("Modelo de Regresión e Interpretación de Impacto")

        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;">
        <strong>¿Qué hace esta sección?</strong><br>
        Aquí usamos matemáticas (Regresión Estadística) para "pesar" las variables. 
        Imagine una balanza: ponemos la <strong>Seguridad</strong> en un lado y la <strong>Educación Financiera</strong> en el otro. 
        El modelo nos dice cuál de las dos inclina más la balanza hacia la decisión de usar la App.
        </div>
        """, unsafe_allow_html=True)
        st.write("") # Espacio

        # --- 1. EJECUCIÓN DEL MODELO ---
        # Preparación de datos
        X = df[["num_seguridad", "num_educacion"]].dropna()

        # Asegurar tipos numéricos
        X = X.apply(pd.to_numeric, errors="coerce").dropna()
        y = df.loc[X.index, "num_intencion"]

        # Añadimos constante solo si hay datos
        if not X.empty:
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            # --- 2. CALIDAD DEL MODELO ---
            st.subheader("1. ¿Podemos confiar en este análisis?")

            col_metric1, col_metric2 = st.columns(2)
            r2 = model.rsquared
            p_f = model.f_pvalue

            with col_metric1:
                st.metric(label="Poder Explicativo (R²)", value=f"{r2:.1%}")
                st.info(f"""
                **Lectura Fácil:** Este modelo explica el **{r2:.1%}** del comportamiento de los usuarios.
                El porcentaje restante depende de cosas que no preguntamos (como tasas de interés, publicidad, necesidad urgente, etc.).
                """)

            with col_metric2:
                st.metric(label="Certeza Estadística (Valor P)", value=f"{p_f:.4f}")
                if p_f < 0.05:
                    st.success("**Veredicto:** El análisis es SÓLIDO. Los resultados no son producto de la suerte.")
                else:
                    st.error("**Veredicto:** Cuidado. Los datos no son suficientes para sacar conclusiones definitivas.")

            st.markdown("---")

            # --- 3. GRÁFICO DE BARRAS (COEFICIENTES) ---
            st.subheader("2. ¿Qué pesa más en la decisión?")
            st.markdown("""
            El siguiente gráfico muestra la **fuerza** de cada variable.
            * **Barra Larga:** Influye mucho.
            * **Barra Verde:** Es un dato confirmado estadísticamente.
            * **Barra Gris:** El dato es incierto (podría ser casualidad).
            """)

            # Diccionario amigable
            nombres_amigables = {
                "const": "Constante (Base)",
                "num_seguridad": "Percepción de Seguridad",
                "num_educacion": "Educación Financiera",
            }
            variables_modelo = [nombres_amigables.get(v, v) for v in model.params.index]

            results_df = pd.DataFrame({
                "Variable": variables_modelo,
                "Impacto": model.params.values,
                "Valor P": model.pvalues.values
            })

            # Filtros para gráfica
            plot_df = results_df[results_df["Variable"] != "Constante (Base)"].copy()
            plot_df["Confiabilidad"] = plot_df["Valor P"].apply(lambda x: "Datos Confiables (Significativo)" if x < 0.05 else "Datos Inciertos (No Sig.)")
            
            # GRÁFICO DE BARRAS SIMPLE
            fig_coef = px.bar(
                plot_df,
                x="Impacto",
                y="Variable",
                color="Confiabilidad",
                orientation="h",
                text_auto='.2f', # Muestra el número en la barra
                color_discrete_map={
                    "Datos Confiables (Significativo)": "#2ecc71", # Verde
                    "Datos Inciertos (No Sig.)": "#bdc3c7"  # Gris
                },
                title="Peso de cada factor en la Intención de Uso"
            )
            
            fig_coef.update_layout(
                xaxis_title="Puntos de Aumento en la Intención de Uso",
                yaxis_title="",
                legend_title="Calidad del Dato",
                template="simple_white",
                font=dict(size=14) # Letra más grande para leer mejor
            )
            st.plotly_chart(fig_coef, use_container_width=True)

            # --- REDACCIÓN AUTOMÁTICA ---
            try:
                coef_seg = model.params["num_seguridad"]
                
                st.markdown(f"""
                <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
                    <strong>Conclusión para el Lector:</strong><br>
                    Por cada punto que usted logre aumentar la <b>Seguridad Percibida</b> en sus usuarios, 
                    la probabilidad de que usen su servicio aumentará en <b>{coef_seg:.2f} puntos</b>.
                    Es decir, invertir en seguridad tiene un retorno directo en la adopción.
                </div>
                """, unsafe_allow_html=True)
            except:
                pass

            st.markdown("---")

            # --- 4. GRÁFICO DE BARRAS (FACTORES) ---
            st.subheader("3. ¿Qué hace que la gente se sienta segura?")
            st.markdown("Analizamos qué palabras clave tienen más relación con una calificación alta de seguridad.")

            factor_cols = [c for c in df.columns if c.startswith("factor_")]
            correlations = {}

            if len(factor_cols) > 0:
                for col in factor_cols:
                    corr_val = df[col].corr(df["num_seguridad"])
                    clean_name = col.replace("factor_", "")
                    correlations[clean_name] = corr_val

                df_corr_factors = pd.DataFrame.from_dict(
                    correlations, orient="index", columns=["Correlación"]
                ).reset_index()
                
                df_corr_factors.columns = ['Factor', 'Correlación']
                df_corr_factors = df_corr_factors.sort_values(by="Correlación", ascending=True) # Ordenar para el gráfico

                # GRÁFICO DE BARRAS DIVERGENTES
                fig_factors = px.bar(
                    df_corr_factors,
                    x="Correlación",
                    y="Factor",
                    orientation='h',
                    color="Correlación",
                    color_continuous_scale="RdBu_r", # Rojo a Azul (Invertido para que Azul sea positivo)
                    title="Impacto de cada característica en la sensación de Seguridad"
                )
                
                fig_factors.add_vline(x=0, line_width=2, line_color="black")
                fig_factors.update_layout(
                    xaxis_title="Relación con la Seguridad (Negativa < 0 > Positiva)",
                    yaxis_title="",
                    template="simple_white"
                )
                st.plotly_chart(fig_factors, use_container_width=True)
                
                st.info("""
                **Cómo leer este gráfico:**
                * Barras hacia la **Derecha (Azul)**: Son los "Ganadores". Mencionar estos factores hace que la gente se sienta más segura.
                * Barras hacia la **Izquierda (Rojo)**: Son factores que, curiosamente, podrían estar asociados a dudas o menor seguridad en esta muestra.
                """)

        else:
            st.error("No hay suficientes datos válidos para generar el modelo.")
else:
    st.info("Esperando archivo CSV para iniciar análisis profesional.")
