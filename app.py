from pathlib import Path
import math

import pandas as pd
import numpy as np
import joblib
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Recomendador de precio Airbnb Madrid",
    layout="centered"
)


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

APP_DIR = Path(__file__).resolve().parent

# Permite ejecutar app.py tanto desde la raíz del proyecto
# como desde una carpeta denominada "app".
PROJECT_PATH = APP_DIR.parent if APP_DIR.name.lower() == "app" else APP_DIR

DATA_PATH = (
    PROJECT_PATH
    / "data"
    / "processed"
    / "model_dataset_ex_ante.csv"
)

MODEL_PATH = (
    PROJECT_PATH
    / "models"
    / "price_model_ex_ante.joblib"
)

FEATURES_PATH = (
    PROJECT_PATH
    / "models"
    / "model_features.joblib"
)


# ============================================================
# CARGA DEL MODELO Y DATOS DE REFERENCIA
# ============================================================

@st.cache_resource
def load_model():
    """
    Carga el pipeline final entrenado y la lista de variables
    esperadas por el modelo.
    """
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, features


@st.cache_data
def load_reference_data():
    """
    Carga el dataset utilizado como referencia para construir
    los desplegables y valores por defecto de la aplicación.
    """
    return pd.read_csv(
        DATA_PATH,
        low_memory=False
    )


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def haversine_km(
    lat,
    lon,
    lat0=40.416775,
    lon0=-3.703790
):
    """
    Calcula la distancia aproximada en kilómetros entre
    el alojamiento y Puerta del Sol mediante la fórmula
    de Haversine.
    """

    r = 6371.0

    p1 = math.radians(lat0)
    p2 = math.radians(lat)

    dp = math.radians(lat - lat0)
    dl = math.radians(lon - lon0)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def mode_or_default(series, default):
    """
    Devuelve la moda de una variable o un valor por defecto
    cuando no existen observaciones válidas.
    """

    s = series.dropna()

    if not s.empty:
        return s.mode().iloc[0]

    return default


# ============================================================
# INICIALIZACIÓN
# ============================================================

model, features = load_model()
df = load_reference_data()


# ============================================================
# CABECERA
# ============================================================

st.title("Recomendador de precio por noche")

st.caption(
    "Prototipo académico · Mercado Airbnb de Madrid · "
    "Precio de referencia, no garantía de ingresos u ocupación"
)


# ============================================================
# LOCALIZACIÓN
# ============================================================
#
# Distrito y barrio se sitúan FUERA del formulario.
#
# Esto permite que Streamlit vuelva a ejecutar la aplicación
# inmediatamente cuando cambia el distrito y actualice así
# automáticamente el listado de barrios.
# ============================================================

st.subheader("Localización del alojamiento")


# ----------------------------
# Distrito
# ----------------------------

districts = sorted(
    df["neighbourhood_group_cleansed"]
    .dropna()
    .astype(str)
    .unique()
)

district = st.selectbox(
    "Distrito",
    districts,
    key="district_selector"
)


# ----------------------------
# Barrios del distrito elegido
# ----------------------------

barrios = sorted(
    df.loc[
        df["neighbourhood_group_cleansed"]
        .astype(str)
        .eq(str(district)),
        "neighbourhood_cleansed"
    ]
    .dropna()
    .astype(str)
    .unique()
)


if len(barrios) == 0:

    st.warning(
        "No se han encontrado barrios para el distrito seleccionado."
    )

    barrios = sorted(
        df["neighbourhood_cleansed"]
        .dropna()
        .astype(str)
        .unique()
    )


neighbourhood = st.selectbox(
    "Barrio",
    barrios,
    key=f"neighbourhood_selector_{district}"
)


# ============================================================
# COORDENADAS REPRESENTATIVAS DEL BARRIO
# ============================================================
#
# Se utilizan como valores iniciales las coordenadas medianas
# observadas para el barrio seleccionado.
# ============================================================

neighbourhood_rows = df.loc[
    df["neighbourhood_cleansed"]
    .astype(str)
    .eq(str(neighbourhood))
]

lat_default = neighbourhood_rows["latitude"].median()
lon_default = neighbourhood_rows["longitude"].median()


if not np.isfinite(lat_default):
    lat_default = 40.4168

if not np.isfinite(lon_default):
    lon_default = -3.7038


# ============================================================
# FORMULARIO DE PREDICCIÓN
# ============================================================

with st.form("pricing_form"):

    # --------------------------------------------------------
    # Características del alojamiento
    # --------------------------------------------------------

    st.subheader("Características del alojamiento")


    property_type = st.selectbox(
        "Tipo de propiedad",
        sorted(
            df["property_type"]
            .dropna()
            .astype(str)
            .unique()
        )
    )


    room_type = st.selectbox(
        "Modalidad",
        sorted(
            df["room_type"]
            .dropna()
            .astype(str)
            .unique()
        )
    )


    # --------------------------------------------------------
    # Capacidad y estructura
    # --------------------------------------------------------

    c1, c2 = st.columns(2)


    with c1:

        accommodates = st.number_input(
            "Capacidad (personas)",
            min_value=1,
            max_value=20,
            value=4
        )

        bedrooms = st.number_input(
            "Dormitorios",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        )

        bathrooms = st.number_input(
            "Baños",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.5
        )


    with c2:

        beds = st.number_input(
            "Camas",
            min_value=0.0,
            max_value=30.0,
            value=2.0,
            step=1.0
        )

        minimum_nights = st.number_input(
            "Estancia mínima (noches)",
            min_value=1.0,
            max_value=365.0,
            value=2.0,
            step=1.0
        )

        maximum_nights = st.number_input(
            "Estancia máxima (noches)",
            min_value=1.0,
            max_value=3650.0,
            value=365.0,
            step=1.0
        )


    # --------------------------------------------------------
    # Coordenadas
    # --------------------------------------------------------

    st.subheader("Localización aproximada")

    c3, c4 = st.columns(2)


    with c3:

        latitude = st.number_input(
            "Latitud",
            value=float(lat_default),
            format="%.6f"
        )


    with c4:

        longitude = st.number_input(
            "Longitud",
            value=float(lon_default),
            format="%.6f"
        )


    # --------------------------------------------------------
    # Equipamientos
    # --------------------------------------------------------

    st.subheader("Equipamientos")


    amenities_count = st.slider(
        "Número total de amenities",
        min_value=0,
        max_value=100,
        value=29
    )


    c5, c6, c7 = st.columns(3)


    with c5:

        has_wifi = st.checkbox(
            "Wifi",
            value=True
        )

        has_kitchen = st.checkbox(
            "Cocina",
            value=True
        )

        has_washer = st.checkbox(
            "Lavadora",
            value=True
        )


    with c6:

        has_air_conditioning = st.checkbox(
            "Aire acondicionado",
            value=True
        )

        has_workspace = st.checkbox(
            "Espacio de trabajo",
            value=False
        )

        has_parking = st.checkbox(
            "Parking",
            value=False
        )


    with c7:

        has_pool = st.checkbox(
            "Piscina",
            value=False
        )

        has_elevator = st.checkbox(
            "Ascensor",
            value=True
        )

        has_balcony_or_patio = st.checkbox(
            "Balcón/patio",
            value=False
        )


    # --------------------------------------------------------
    # Características del anfitrión
    # --------------------------------------------------------

    with st.expander(
        "Características del anfitrión (opcionales)"
    ):

        host_is_superhost = st.selectbox(
            "Superhost",
            ["f", "t"]
        )

        host_experience_months = st.number_input(
            "Experiencia como host (meses)",
            min_value=0.0,
            max_value=300.0,
            value=36.0,
            step=1.0
        )

        host_listings_count = st.number_input(
            "Número de anuncios del host",
            min_value=0.0,
            max_value=10000.0,
            value=1.0,
            step=1.0
        )

        host_has_profile_pic = st.selectbox(
            "Tiene foto de perfil",
            ["t", "f"]
        )

        host_identity_verified = st.selectbox(
            "Identidad verificada",
            ["t", "f"]
        )

        has_license_info = int(
            st.checkbox(
                "Existe información de licencia",
                value=True
            )
        )


    # --------------------------------------------------------
    # Botón de predicción
    # --------------------------------------------------------

    submitted = st.form_submit_button(
        "Estimar precio"
    )


# ============================================================
# GENERACIÓN DE LA PREDICCIÓN
# ============================================================

if submitted:

    # --------------------------------------------------------
    # Valores iniciales
    # --------------------------------------------------------
    #
    # Para aquellas variables requeridas por el modelo que no
    # aparecen explícitamente en la interfaz, se utilizan valores
    # representativos del dataset de referencia:
    #
    # - mediana para variables numéricas
    # - moda para variables categóricas
    # --------------------------------------------------------

    row = {}


    for col in features:

        if col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):

                median_value = df[col].median()

                row[col] = (
                    float(median_value)
                    if pd.notna(median_value)
                    else np.nan
                )

            else:

                row[col] = mode_or_default(
                    df[col],
                    np.nan
                )

        else:

            row[col] = np.nan


    # --------------------------------------------------------
    # Valores introducidos por el usuario
    # --------------------------------------------------------

    row.update({

        "neighbourhood_cleansed":
            neighbourhood,

        "neighbourhood_group_cleansed":
            district,

        "latitude":
            latitude,

        "longitude":
            longitude,

        "distance_to_sol_km":
            haversine_km(
                latitude,
                longitude
            ),

        "property_type":
            property_type,

        "room_type":
            room_type,

        "accommodates":
            accommodates,

        "bathrooms":
            bathrooms,

        "bedrooms":
            bedrooms,

        "beds":
            beds,

        "minimum_nights":
            minimum_nights,

        "maximum_nights":
            maximum_nights,

        "host_is_superhost":
            host_is_superhost,

        "host_listings_count":
            host_listings_count,

        "host_has_profile_pic":
            host_has_profile_pic,

        "host_identity_verified":
            host_identity_verified,

        "host_experience_months":
            host_experience_months,

        "amenities_count":
            amenities_count,

        "has_license_info":
            has_license_info,

        "has_wifi":
            int(has_wifi),

        "has_air_conditioning":
            int(has_air_conditioning),

        "has_kitchen":
            int(has_kitchen),

        "has_washer":
            int(has_washer),

        "has_workspace":
            int(has_workspace),

        "has_parking":
            int(has_parking),

        "has_pool":
            int(has_pool),

        "has_elevator":
            int(has_elevator),

        "has_balcony_or_patio":
            int(has_balcony_or_patio)
    })


    # --------------------------------------------------------
    # Construcción del registro esperado por el modelo
    # --------------------------------------------------------

    X_new = pd.DataFrame(
        [row]
    ).reindex(
        columns=features
    )


    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    pred = float(
        model.predict(X_new)[0]
    )


    # ========================================================
    # RESULTADO
    # ========================================================

    st.success(
        "Estimación realizada correctamente"
    )


    st.metric(
        "Precio de referencia estimado",
        f"{pred:,.2f} € / noche"
    )


    st.write(
        "Distancia aproximada a Puerta del Sol: "
        f"**{row['distance_to_sol_km']:.2f} km**"
    )


    st.info(
        "La estimación debe interpretarse como una referencia "
        "de mercado basada en patrones observados. "
        "No constituye una garantía de maximización de "
        "ocupación, ingresos o rentabilidad."
    )
