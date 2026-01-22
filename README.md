# Repositorio de Analítica de Negocios

Este repositorio contiene una colección de notebooks de Jupyter diseñados como material de estudio para cursos de analítica de negocios y machine learning. Los ejemplos están pensados para ser ejecutados en Google Colab, facilitando el acceso y la experimentación sin necesidad de configuración local.

## Temas

El repositorio está organizado en los siguientes temas, cubriendo desde conceptos fundamentales hasta aplicaciones más avanzadas:

1.  [Clasificadores Naive Bayes](#bayes)
2.  [Árboles de Decisión](#árboles-de-decisión)
3.  [Clustering con K-Means](#clustering-k-means)
4.  [Redes Neuronales con PyTorch](#redes-neuronales)
5.  [Procesamiento de Lenguaje Natural (NLP)](#procesamiento-de-lenguaje-natural-nlp)

---

### Bayes
En esta sección se introducen los clasificadores Naive Bayes, una familia de algoritmos de clasificación probabilísticos simples basados en el teorema de Bayes con fuertes (ingenuas) suposiciones de independencia entre las características.

| Notebook | Descripción |
| --- | --- |
| **01 - bayes.ipynb** | Introduce el clasificador Naive Bayes con un ejemplo de clasificación de texto. Se implementa un modelo Multinomial Naive Bayes desde cero para clasificar frases como "spam" o "normal". |
| **02 - bayes gaussiano.ipynb** | Aplica un clasificador Naive Bayes Gaussiano a un problema de scoring de crédito. El notebook carga datos de solicitantes, visualiza las distribuciones de las características y entrena un modelo para predecir la pre-aprobación de un crédito, evaluando su rendimiento con varias métricas. |

---

### Árboles de Decisión
Aquí se exploran los árboles de decisión, un tipo de modelo de aprendizaje supervisado no paramétrico utilizado para clasificación y regresión.

| Notebook | Descripción |
| --- | --- |
| **01 - arboles.ipynb** | Construye e interpreta un clasificador de Árbol de Decisión para una aplicación de riesgo de crédito. Se visualiza el árbol resultante para entender sus reglas de decisión y se evalúa su rendimiento con una matriz de confusión y otras métricas. |

---

### Clustering (K-Means)
Esta sección se centra en el clustering, una tarea de aprendizaje no supervisado, utilizando el popular algoritmo K-Means para agrupar datos no etiquetados.

| Notebook | Descripción |
| --- | --- |
| **01 - Clustering (K_means).ipynb** | Aplica el algoritmo K-Means para segmentar solicitantes de crédito en diferentes perfiles. El notebook realiza un análisis exploratorio, implementa K-Means para agrupar a los solicitantes y visualiza los clusters resultantes. |

---

### Redes Neuronales
Introducción a las redes neuronales utilizando PyTorch, una de las bibliotecas de aprendizaje profundo más populares.

| Notebook | Descripción |
| --- | --- |
| **00 - tensores.ipynb** | Una introducción básica a los tensores de PyTorch, los bloques de construcción fundamentales de las redes neuronales. Se cubren varias formas de crear y operar con tensores. |
| **01 - FirstNL.ipynb** | Guía para construir una primera red neuronal simple para una tarea de regresión. Se demuestra un flujo de trabajo completo, incluyendo la preparación y estandarización de datos, la definición del modelo y el bucle de entrenamiento. |
| **02 - Madeline.ipynb** | Aborda un problema de regresión no lineal, introduciendo conceptos clave como las capas ocultas y las funciones de activación (ReLU) para construir modelos más potentes capaces de aprender relaciones complejas. |
| **03 - credito.ipynb** | Aplica una red neuronal a un problema de clasificación del mundo real: la predicción de la aprobación de créditos. Este es un ejemplo completo que abarca desde la carga y preprocesamiento de datos hasta el entrenamiento y la evaluación del modelo. |

---

### Procesamiento de Lenguaje Natural (NLP)
Esta sección explora técnicas para que las computadoras entiendan y procesen el lenguaje humano.

| Notebook | Descripción |
| --- | --- |
| **01 - NLP clasico.ipynb** | Introduce técnicas fundamentales de NLP "clásico" como la tokenización, normalización de texto (minúsculas, eliminación de puntuación y stopwords) y la extracción de características con TF-IDF. |
| **02 - modelos preentrenados.ipynb** | Demuestra el poder de los modelos de embeddings de palabras pre-entrenados (Glove) para capturar relaciones semánticas. También muestra cómo usar un modelo de Transformers pre-entrenado para el análisis de sentimientos. |
| **03 - Few-Shot** | Introduce el concepto de "Few-Shot Learning" usando un modelo de IA generativa (Gemini). En lugar de un entrenamiento completo, se proporcionan algunos ejemplos en el prompt para guiar al modelo en la clasificación de reseñas de clientes. |
