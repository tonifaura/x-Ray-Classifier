import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import time

# Estilos personalizados
st.markdown(
    """
    <style>
    .uploadedFileName, .uploadedFileSize {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True
)

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    return load_model("xrayclassifier.keras")

modelo = cargar_modelo()

# Clases del modelo
class_names = [
    "Abdomen", "Tobillo", "Columna cervical", "Tórax", "Clavículas",
    "Codo", "Pies", "Dedos", "Antebrazo", "Mano", "Cadera", "Rodilla",
    "Pierna", "Columna lumbar", "Otros", "Pelvis", "Hombro", "Senos paranasales",
    "Cráneo", "Muslo", "Columna torácica", "Muñeca"
]

# Inicializar st.session_state
if "results" not in st.session_state:
    st.session_state.results = []  # Lista para historial
    st.session_state.current = None  # Resultado actual

# Preprocesar imagen
def preprocess_image(image, target_size=(256, 256)):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Generar Grad-CAM++
def grad_cam_plus_plus(model, img_array, class_idx):
    last_conv_layer = model.get_layer("conv5_block30_concat")  # Ajusta según tu modelo
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    grads_squared = tf.square(grads)
    grads_cubed = grads_squared * grads

    conv_outputs_np = conv_outputs[0].numpy()
    grads_np = grads[0].numpy()
    grads_squared_np = grads_squared[0].numpy()
    grads_cubed_np = grads_cubed[0].numpy()

    numerator = grads_cubed_np
    denominator = (2.0 * grads_cubed_np + grads_squared_np * np.sum(conv_outputs_np, axis=(0, 1)) + 1e-8)
    weights = np.sum(numerator / denominator, axis=(0, 1))

    # Calcular el mapa de activación
    activation_map = np.zeros(conv_outputs_np.shape[:2], dtype=np.float32)
    for i in range(conv_outputs_np.shape[-1]):
        activation_map += weights[i] * conv_outputs_np[:, :, i]

    # Normalización del mapa
    activation_map = np.maximum(activation_map, 0)
    activation_map = cv2.resize(activation_map, (256, 256))
    activation_map = activation_map / np.max(activation_map)
    activation_map = 1.0 - activation_map

    return activation_map

# Mostrar Grad-CAM++
def mostrar_grad_cam(image, heatmap):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    st.image(superimposed_img, caption="Grad-CAM++ sobre la imagen", use_column_width=True)

# Mostrar logo en la barra lateral con tamaño ajustado
st.sidebar.image("logo.png", width=265)

# Subir imágenes
st.title("Clasificación de radiografías según la parte del cuerpo")
st.write("Sube una o más imágenes")

uploaded_files = st.file_uploader(
    " ", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    num_files = len(uploaded_files)

    # Agregar botón en la barra lateral para mostrar/ocultar Grad-CAM en todas las imágenes
    st.sidebar.header("Control Global")
    activar_todos_gradcam = st.sidebar.button("Activar Grad-CAM++ para todas las imágenes")
    desactivar_todos_gradcam = st.sidebar.button("Desactivar Grad-CAM++ para todas las imágenes")

    if activar_todos_gradcam:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state:
                st.session_state[uploaded_file.name]["show_gradcam"] = True
        st.experimental_rerun()

    if desactivar_todos_gradcam:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state:
                st.session_state[uploaded_file.name]["show_gradcam"] = False
        st.experimental_rerun()

    # Botón para ocultar todas las imágenes de la página principal
    ocultar_todo = st.sidebar.button("Ocultar todas las imágenes")
    if ocultar_todo:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state:
                st.session_state[uploaded_file.name]["visible"] = False
        st.experimental_rerun()

    # Si hay más de una imagen, usar diseño de cuadrícula
    if num_files > 1:
        # Crear una barra de progreso global antes de empezar a procesar las imágenes
        progress = st.progress(0)
        total_images = len(uploaded_files)
        progress_increment = 100 / (total_images * 4)  # Incremento de progreso por cada etapa de cada imagen

        cols = st.columns(3)  # Máximo 3 imágenes por fila
        col_indices = [0, 1, 2]  # Índices de las columnas disponibles

        # Crear una lista para rastrear posiciones ocupadas en la cuadrícula
        posiciones_ocupadas = [False, False, False]

        for uploaded_file in uploaded_files:
            # Comprobar si la imagen ya está procesada
            if uploaded_file.name not in st.session_state:
                image = Image.open(uploaded_file)

                # Procesar la imagen (sin barra de progreso individual)
                img_array = preprocess_image(image)
                # Actualizar el progreso global tras el preprocesamiento
                progress.progress(int((uploaded_files.index(uploaded_file) * 4 + 1) * progress_increment))

                # Realizar la predicción
                predicciones = modelo.predict(img_array)
                # Actualizar el progreso global tras la predicción
                progress.progress(int((uploaded_files.index(uploaded_file) * 4 + 2) * progress_increment))

                clase_predicha = np.argmax(predicciones)
                confianza = np.max(predicciones)

                # Generar el heatmap
                heatmap = grad_cam_plus_plus(modelo, img_array, clase_predicha)
                # Actualizar el progreso global tras la generación del heatmap
                progress.progress(int((uploaded_files.index(uploaded_file) * 4 + 3) * progress_increment))

                # Guardar la imagen procesada en el estado de sesión
                st.session_state[uploaded_file.name] = {
                    "image": image,
                    "prediction": class_names[clase_predicha],
                    "confidence": confianza,
                    "heatmap": heatmap,
                    "visible": True,
                    "show_gradcam": False,
                }

                # Actualizar la barra de progreso global al 100% después de completar
                progress.progress(int((uploaded_files.index(uploaded_file) + 1) * progress_increment * 4))

            # Obtener resultados de la sesión
            result = st.session_state[uploaded_file.name]

            # Mostrar las imágenes en la cuadrícula
            if result["visible"]:
                # Buscar la primera posición libre en la cuadrícula
                for idx in col_indices:
                    if not posiciones_ocupadas[idx]:
                        col = cols[idx]  # Seleccionar la columna
                        posiciones_ocupadas[idx] = True  # Marcar como ocupada
                        break

                with col:
                    st.image(result["image"], caption="Imagen cargada", use_column_width=True)
                    st.subheader(f"Parte del cuerpo: {result['prediction']}")
                    color = "green" if result["confidence"] > 0.95 else "orange" if result["confidence"] > 0.75 else "red"
                    st.markdown(
                        f"<p style='color:{color};font-size:18px;'>Confianza: {result['confidence']:.2%}</p>",
                        unsafe_allow_html=True,
                    )

                    # Botón para mostrar Grad-CAM++
                    if st.button("Mostrar Grad-CAM++", key=f"gradcam_{uploaded_file.name}"):
                        result["show_gradcam"] = not result["show_gradcam"]
                        st.experimental_rerun()

                    if result["show_gradcam"]:
                        mostrar_grad_cam(result["image"], result["heatmap"])

                    # Botón para ocultar la imagen
                    if st.button(f"Ocultar", key=f"ocultar_{uploaded_file.name}"):
                        st.session_state[uploaded_file.name]["visible"] = False
                        st.experimental_rerun()

    else:
        # Si hay solo una imagen, mostrar en vista expandida
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)

            if uploaded_file.name not in st.session_state:
                progreso = st.progress(0)

                img_array = preprocess_image(image)
                progreso.progress(25)

                predicciones = modelo.predict(img_array)
                progreso.progress(50)

                clase_predicha = np.argmax(predicciones)
                confianza = np.max(predicciones)

                progreso.progress(75)

                heatmap = grad_cam_plus_plus(modelo, img_array, clase_predicha)
                progreso.progress(100)

                st.session_state[uploaded_file.name] = {
                    "image": image,
                    "prediction": class_names[clase_predicha],
                    "confidence": confianza,
                    "heatmap": heatmap,
                    "visible": True,
                    "show_gradcam": False,
                }

            result = st.session_state[uploaded_file.name]

            if result["visible"]:
                with st.expander(f"Imagen procesada", expanded=True):
                    st.image(result["image"], caption="Imagen cargada", use_column_width=True)
                    st.subheader(f"Parte del cuerpo: {result['prediction']}")
                    color = "green" if result["confidence"] > 0.8 else "orange" if result["confidence"] > 0.5 else "red"
                    st.markdown(
                        f"<p style='color:{color};font-size:18px;'>Confianza: {result['confidence']:.2%}</p>",
                        unsafe_allow_html=True,
                    )

                    if st.button("Mostrar Grad-CAM++", key=f"gradcam_{uploaded_file.name}"):
                        result["show_gradcam"] = not result["show_gradcam"]
                        st.experimental_rerun()

                    if result["show_gradcam"]:
                        mostrar_grad_cam(result["image"], result["heatmap"])

                    ocultar = st.button(
                        f"Ocultar de la página principal", key=f"ocultar_{uploaded_file.name}"
                    )
                    if ocultar:
                        st.session_state[uploaded_file.name]["visible"] = False
                        st.experimental_rerun()

# Mostrar historial en la barra lateral
st.sidebar.header("Historial")
for uploaded_file in uploaded_files:
    if uploaded_file.name in st.session_state:
        result = st.session_state[uploaded_file.name]
        st.sidebar.image(result["image"], caption=f"{result['prediction']}", width=100)