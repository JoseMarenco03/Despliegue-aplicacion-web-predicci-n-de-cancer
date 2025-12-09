import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Cargar el modelo y el escalador pre-entrenados
scaler = joblib.load('scaler.joblib')
model = joblib.load('ensemble_voting_original_model.joblib')

# Columnas originales de X antes de eliminar las correlacionadas
original_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se',
    'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
    'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# Columnas a eliminar para el preprocesamiento (mismas que se eliminaron en el entrenamiento)
columns_to_drop = ['texture_worst', 'concavity_se', 'fractal_dimension_se']

st.set_page_config(page_title="Predicción de Cáncer de Mama")
st.title("Aplicación de Predicción de Cáncer de Mama")
st.write("Ingrese los valores de las características para predecir el diagnóstico.")

# 2. Crear campos de entrada para el usuario
input_data = {}
for feature in original_features:
    input_data[feature] = st.number_input(
        f"Ingrese el valor para {feature}",
        value=0.5,
        format="%.4f"
    )

# Botón de predicción
if st.button('Realizar Predicción'):
    # 3. Preparar los datos de entrada del usuario
    # Aseguramos el MISMO orden de columnas que en el entrenamiento
    input_df = pd.DataFrame([input_data])[original_features]

    # Eliminar las mismas columnas que se eliminaron durante el entrenamiento
    input_df_dropped = input_df.drop(columns=columns_to_drop, errors='ignore')

    # Si el scaler fue entrenado con nombres de columnas, respetamos ese orden
    if hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
        input_df_dropped = input_df_dropped[expected_cols]

    # Escalar los datos de entrada usando el mismo escalador
    scaled_input = scaler.transform(input_df_dropped)

    # 4. Realizar la predicción
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    # 5. Mostrar el resultado
    st.subheader("Resultado de la Predicción:")
    if prediction[0] == 0:
        st.success("El diagnóstico predicho es: **Benigno**")
    else:
        st.error("El diagnóstico predicho es: **Maligno**")

    st.write(f"Probabilidad de Benigno: {prediction_proba[0][0]:.4f}")
    st.write(f"Probabilidad de Maligno: {prediction_proba[0][1]:.4f}")

st.markdown("""
---
#### Cómo ejecutar esta aplicación:
1. Guarda el código de arriba en un archivo llamado `app.py` en tu entorno local.
2. Asegúrate de que los archivos `scaler.joblib` y `ensemble_voting_original_model.joblib` estén en la misma carpeta.
3. Abre tu terminal, navega a esa carpeta y ejecuta: `streamlit run app.py`
""")

