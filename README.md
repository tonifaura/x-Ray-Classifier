# WebApp de Clasificación de Radiografías con IA Interpretable
Este proyecto es una webapp que clasifica radiografías según la parte del cuerpo utilizando un modelo de deep learning. Además, incorpora herramientas de explicabilidad como Grad-CAM++ y niveles de confianza para mejorar la interpretabilidad de las predicciones.

### Demo
[![Video en YouTube](https://img.youtube.com/vi/1CS3Zk7abdE/maxresdefault.jpg)](https://youtu.be/1CS3Zk7abdE?si=pC-3P6x4yg-vsDty)

---

## Índice
0. [Demo](#demo)
1. [Introducción y Motivación](#introducción-y-motivación)
3. [Objetivos del Proyecto](#objetivos-del-proyecto)
4. [Conjunto de Datos](#conjunto-de-datos)
5. [Preprocesamiento y Limpieza](#preprocesamiento-y-limpieza)
6. [Aumento de Datos](#aumento-de-datos)
7. [Modelo Utilizado](#modelo-utilizado)
9. [Resultados](#resultados)
10. [Explicabilidad](#explicabilidad)
11. [Deployment](#deployment)
12. [Conclusiones y Próximos Pasos](#conclusiones-y-próximos-pasos)

---

## Introducción y Motivación

Este proyecto nació como un reto del máster, en el que junto a dos compañeros, trabajamos con un dataset de +1.200 radiografías para entrenar un modelo de clasificación.
Posteriormente decidí retomarlo y llevarlo un paso más allá, mejorando el modelo, la interpretabilidad y la visualización de las predicciones.

---

## Objetivos del Proyecto

1. **Precisión de Clasificación**: Asegurar una alta precisión en la categorización de imágenes.
2. **Eficiencia Computacional**: Mejorar la velocidad de procesamiento para una mayor escalabilidad.
3. **Confianza en las Predicciones**: Proporcionar un indicador de confianza para cada clasificación, permitiendo evaluar la fiabilidad del modelo.
4. **Interpretabilidad**: Implementar técnicas de explicabilidad para visualizar qué zonas de la imagen han influido en la predicción.

---

## Conjunto de Datos

- **Cantidad**: 1278 imágenes de entrenamiento y 328 de prueba.
- **Formato**: Imágenes en formato DICOM.
- **Etiquetas**:

   Abdomen = 0

   Tobillo = 1

   Columna cervical = 2

   Tórax = 3

   Clavículas = 4

   Codo = 5

   Pies = 6

   Dedos = 7

   Antebrazo = 8

   Mano = 9

   Cadera = 10

   Rodilla = 11

   Pierna = 12

   Columna lumbar = 13

   Otros = 14

   Pelvis = 15

   Hombro = 16

   Senos paranasales = 17

   Cráneo = 18

   Muslo = 19

   Columna torácica = 20

   Muñeca = 21

- **Características Adicionales**: Cada imagen cuenta con un `SOPInstanceUID` único y una columna `target` para la categoría.

---

## Preprocesamiento y Limpieza

- **Tamaño de Imagen**: Redimensionado a 256x256 píxeles.
- **Formatos**: Conversión de DICOM a PNG.
- **Escala de Color**: Escala monocromática (`Monochrome2`) con normalización de valores entre -1 y 1.
- **Limpieza de Dataset**: Eliminación de imágenes irrelevantes, como aquellas que no correspondían a radiografías válidas.

---

## Aumento de Datos

Para mejorar la variabilidad del conjunto de entrenamiento, se aplicaron las siguientes técnicas de aumento de datos:

- **Rotaciones**: Hasta 20 grados, además de rotaciones de 90 grados.
- **Traslaciones**: Horizontales y verticales.
- **Recortes (shear)**: Modificaciones en la forma de la imagen.
- **Zoom**: Simulación de acercamientos y alejamientos.

---

## Modelo Utilizado

Este modelo se basa en la arquitectura **DenseNet201** de Keras, preentrenada con los pesos de ImageNet. 

El modelo utiliza un enfoque de transfer learning, donde las primeras capas de DenseNet201 se mantienen congeladas para aprovechar las características aprendidas previamente, mientras que las capas posteriores son ajustadas para la tarea específica. Se añaden capas adicionales de procesamiento, incluyendo un GlobalAveragePooling2D seguido de una capa Dense con 1024 unidades y una capa final de clasificación con 22 salidas, usando softmax para la clasificación multiclase. El modelo es optimizado con el optimizador Adam y la función de pérdida sparse_categorical_crossentropy.

- **Tamaño del modelo**: 14 MB
- **Parámetros**: 32 M
- **Profundidad**: 710 capas

---

## Resultados

| Conjunto       | Accuracy | Loss   |
|----------------|----------|--------|
| Entrenamiento  | 0.9777   | 0.0669 |
| Validación     | 0.9727   | 0.1454 |

*_epoch 12._

---

## Explicabilidad

### Confianza en las Predicciones
Para evaluar qué tan segura es cada clasificación, el modelo usa Softmax, una función que convierte las predicciones en probabilidades. La confianza es simplemente la probabilidad más alta entre todas las clases posibles.

- Sistema de colores en la webapp: 

🟢 Verde → Confianza > 90% (Predicción muy segura)

🟠 Naranja → Confianza entre 70%-90% (Predicción razonable)

🔴 Rojo → Confianza < 70% (Predicción incierta, posible error del modelo)

Este enfoque permite a los usuarios evaluar cuándo confiar en la predicción y cuándo tomarla con cautela.


### Visualización con Grad-CAM++
Para mejorar la interpretabilidad, la app incorpora Grad-CAM++ en la última capa convolucional de la red. Esta técnica genera mapas de calor que muestran qué regiones de la imagen activaron más el modelo en su predicción.

¿Por qué es útil?

Permite verificar si el modelo se está fijando en la región correcta.
Ayuda a detectar sesgos (ej. si clasifica por bordes o texto en la imagen en lugar de la anatomía).
Refuerza la confianza del usuario al mostrar cómo el modelo tomó la decisión.

Capa utilizada: En este modelo, la visualización con Grad-CAM++ se genera a partir de la activación de la capa conv5_block30_concat de DenseNet201, en lugar de la última (conv5_block32_concat). Esto permite obtener mapas de calor más interpretables en este caso específico.

---

## Deployment

 Características de la webapp:
 - Subida de imágenes para clasificación inmediata.
- Historial de predicciones para revisar y comparar resultados previos.
- Modo de visualización flexible: permite analizar imágenes de una en una en tamaño grande o procesar múltiples imágenes simultáneamente, simulando un flujo de trabajo real en entornos médicos.
- Interfaz optimizada con cuadrículas y controles globales en la barra lateral para una navegación eficiente.

---

## Conclusiones y Próximos Pasos

Este proyecto demuestra que la interpretabilidad es clave en cualquier solución de IA médica, incluso en problemas sencillos como la clasificación de radiografías por parte del cuerpo. La incorporación de confianza en las predicciones y visualización con Grad-CAM++ permite a los usuarios entender mejor el modelo y evaluar su fiabilidad.

Además, utilizar DenseNet201 con Transfer Learning ha permitido obtener buenos resultados con un conjunto de datos limitado (1.200 imágenes), acelerando el entrenamiento y mejorando la generalización.

1. **Ampliación del dataset**: Aumentar la cantidad y diversidad de imágenes para mejorar la robustez del modelo, o para poder usar un modelo menos pesado.
2. **Explicabilidad avanzada**: En lugar de un único mapa de calor, generar una animación (GIF) con la progresión de activaciones a través de diferentes capas/bloques del modelo. Esto ayudaría a visualizar mejor cómo el modelo procesa la imagen.
3. **Aplicación en Detección de Anomalías**: Explorar la integración de técnicas de detección de anomalías para identificar posibles fracturas o patologías.
4. **Implementación de un backend**: Mejorar la escalabilidad y permitir el uso de la API por otras aplicaciones.
5. **Mejor experiencia de usuario**
