from pathlib import Path
import math
import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title='Recomendador de precio Airbnb Madrid', layout='centered')

APP_DIR = Path(__file__).resolve().parent
# El fichero puede colocarse dentro de TFM/app o directamente en la raíz de TFM.
PROJECT_PATH = APP_DIR.parent if APP_DIR.name.lower() == 'app' else APP_DIR
DATA_PATH = PROJECT_PATH / 'data' / 'processed' / 'model_dataset_ex_ante.csv'
MODEL_PATH = PROJECT_PATH / 'models' / 'price_model_ex_ante.joblib'
FEATURES_PATH = PROJECT_PATH / 'models' / 'model_features.joblib'

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)

@st.cache_data
def load_reference_data():
    return pd.read_csv(DATA_PATH, low_memory=False)


def haversine_km(lat, lon, lat0=40.416775, lon0=-3.703790):
    r = 6371.0
    p1, p2 = math.radians(lat0), math.radians(lat)
    dp = math.radians(lat-lat0)
    dl = math.radians(lon-lon0)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(math.sqrt(a))


def mode_or_default(series, default):
    s = series.dropna()
    return s.mode().iloc[0] if not s.empty else default

model, features = load_model()
df = load_reference_data()

st.title('Recomendador de precio por noche')
st.caption('Prototipo académico · Mercado Airbnb de Madrid · Precio de referencia, no garantía de ingresos u ocupación')

with st.form('pricing_form'):
    st.subheader('Características del alojamiento')
    # Listado de distritos disponibles
districts = sorted(
    df["neighbourhood_group_cleansed"]
    .dropna()
    .astype(str)
    .unique()
)

# Selector de distrito
district = st.selectbox(
    "Distrito",
    districts,
    key="district"
)

# Los barrios se recalculan en función del distrito seleccionado
barrios = sorted(
    df.loc[
        df["neighbourhood_group_cleansed"].astype(str) == district,
        "neighbourhood_cleansed"
    ]
    .dropna()
    .astype(str)
    .unique()
)

# Selector dependiente de barrio
neighbourhood = st.selectbox(
    "Barrio",
    barrios,
    key=f"neighbourhood_{district}"
)

    property_type = st.selectbox('Tipo de propiedad', sorted(df['property_type'].dropna().astype(str).unique()))
    room_type = st.selectbox('Modalidad', sorted(df['room_type'].dropna().astype(str).unique()))

    c1, c2 = st.columns(2)
    with c1:
        accommodates = st.number_input('Capacidad (personas)', 1, 20, 4)
        bedrooms = st.number_input('Dormitorios', 0.0, 20.0, 2.0, 0.5)
        bathrooms = st.number_input('Baños', 0.0, 20.0, 1.0, 0.5)
    with c2:
        beds = st.number_input('Camas', 0.0, 30.0, 2.0, 1.0)
        minimum_nights = st.number_input('Estancia mínima (noches)', 1.0, 365.0, 2.0, 1.0)
        maximum_nights = st.number_input('Estancia máxima (noches)', 1.0, 3650.0, 365.0, 1.0)

    st.subheader('Localización aproximada')
    lat_default = float(df.loc[df['neighbourhood_cleansed'].astype(str)==neighbourhood, 'latitude'].median())
    lon_default = float(df.loc[df['neighbourhood_cleansed'].astype(str)==neighbourhood, 'longitude'].median())
    if not np.isfinite(lat_default): lat_default = 40.4168
    if not np.isfinite(lon_default): lon_default = -3.7038
    c3, c4 = st.columns(2)
    with c3:
        latitude = st.number_input('Latitud', value=lat_default, format='%.6f')
    with c4:
        longitude = st.number_input('Longitud', value=lon_default, format='%.6f')

    st.subheader('Equipamientos')
    amenities_count = st.slider('Número total de amenities', 0, 100, 29)
    c5, c6, c7 = st.columns(3)
    with c5:
        has_wifi = st.checkbox('Wifi', True)
        has_kitchen = st.checkbox('Cocina', True)
        has_washer = st.checkbox('Lavadora', True)
    with c6:
        has_air_conditioning = st.checkbox('Aire acondicionado', True)
        has_workspace = st.checkbox('Espacio de trabajo', False)
        has_parking = st.checkbox('Parking', False)
    with c7:
        has_pool = st.checkbox('Piscina', False)
        has_elevator = st.checkbox('Ascensor', True)
        has_balcony_or_patio = st.checkbox('Balcón/patio', False)

    with st.expander('Características del anfitrión (opcionales)'):
        host_is_superhost = st.selectbox('Superhost', ['f','t'])
        host_experience_months = st.number_input('Experiencia como host (meses)', 0.0, 300.0, 36.0, 1.0)
        host_listings_count = st.number_input('Número de anuncios del host', 0.0, 10000.0, 1.0, 1.0)
        host_has_profile_pic = st.selectbox('Tiene foto de perfil', ['t','f'])
        host_identity_verified = st.selectbox('Identidad verificada', ['t','f'])
        has_license_info = int(st.checkbox('Existe información de licencia', True))

    submitted = st.form_submit_button('Estimar precio')

if submitted:
    # Partimos de valores representativos para las variables menos relevantes que no se muestran en la interfaz.
    row = {}
    for col in features:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                row[col] = float(df[col].median()) if pd.notna(df[col].median()) else np.nan
            else:
                row[col] = mode_or_default(df[col], np.nan)
        else:
            row[col] = np.nan

    row.update({
        'neighbourhood_cleansed': neighbourhood,
        'neighbourhood_group_cleansed': district,
        'latitude': latitude,
        'longitude': longitude,
        'distance_to_sol_km': haversine_km(latitude, longitude),
        'property_type': property_type,
        'room_type': room_type,
        'accommodates': accommodates,
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'beds': beds,
        'minimum_nights': minimum_nights,
        'maximum_nights': maximum_nights,
        'host_is_superhost': host_is_superhost,
        'host_listings_count': host_listings_count,
        'host_has_profile_pic': host_has_profile_pic,
        'host_identity_verified': host_identity_verified,
        'host_experience_months': host_experience_months,
        'amenities_count': amenities_count,
        'has_license_info': has_license_info,
        'has_wifi': int(has_wifi),
        'has_air_conditioning': int(has_air_conditioning),
        'has_kitchen': int(has_kitchen),
        'has_washer': int(has_washer),
        'has_workspace': int(has_workspace),
        'has_parking': int(has_parking),
        'has_pool': int(has_pool),
        'has_elevator': int(has_elevator),
        'has_balcony_or_patio': int(has_balcony_or_patio),
    })

    X_new = pd.DataFrame([row]).reindex(columns=features)
    pred = float(model.predict(X_new)[0])

    st.metric('Precio de referencia estimado', f'{pred:,.2f} € / noche')
    st.write(f'Distancia aproximada a Puerta del Sol: **{row["distance_to_sol_km"]:.2f} km**')
    st.info('La estimación debe interpretarse como referencia de mercado basada en patrones observados. No constituye una garantía de maximización de ocupación, ingresos o rentabilidad.')
