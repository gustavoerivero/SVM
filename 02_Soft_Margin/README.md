# SVM de Margen Suave (Soft Margin) con Núcleos (Kernels)

---

Este repositorio contiene la implementación académica de una Máquina de Vectores de Soporte (SVM) de Margen Suave, diseñada para resolver problemas de clasificación no lineal mediante el uso de espacios de características inducidos por Kernels.

## 1. Estructura del código

Este proyecto está organizado para facilitar la trazabilidad de los experimentos.

* ``src/``: Contiene el código fuente principal ([main.py](./src/main.py "Código fuente"))
* ``data/``: Almacena los resultados tabulados en formato CSV ([reporte_comparativo_kernels.csv](./data/reporte_comparativo_kernels.csv))
* ``images/``: Contiene la visualización de las fronteras de decisión y los vectores de soporte.

## 2. Guía de Ejecución

### Requisitos Previos

Asegúrese de tener instalado el entorno de Python con las bibliotecas necesarias:

```bash
pip install numpy pandas scipy matplotlib scikit-learn
```

### Ejecución

Para procesar el conjunto de datos de arquetipos y generar el reporte comparativo, ejecute:

```bash
cd src
python main.py
```

## 3. Descripción del Código

El sistema está diseñado modularmente para garantizar la estabilidad y la reproducibilidad:

### ``KernelSVM`` (Clase)

Implementa la optimización dual del SVM.

* ``fit()``: Resuelve el problema cuadrático mediante el método ``trust-constr``. La función objetivo a minimizar es:

$$
\text{Obj}(\alpha) = \frac{1}{2} \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j K(x_i, x_j) - \sum_{i} \alpha_i
$$

* ``decision_func()``: Evalúa la función de decisión para una instancia dada:

$$
f(x) = \sum_{i} (\alpha_i y_i K(x_i, x)) + b
$$

### Funciones de Kernel

Implementaciones matemáticas de los núcleos:

| Kernel     | Expresión                                            |
| ---------- | ----------------------------------------------------- |
| Lineal     | **$K(x, y) = x^T y$**                         |
| Polinomial | **$K(x, y) = (x^T y + c)^d$**                 |
| RBF        | **$K(x, y) = \exp(-\gamma \|x-y\|^2)$**       |
| Sigmoide   | **$K(x, y) = \tanh(\gamma(x^T y) + \theta)$** |

## 4. Análisis de Resultados

El experimento comparativo evaluó 15 configuraciones diferentes variando el tipo de Kernel y el parámetro de penalización ``C``.

### Interpretación de métricas

La Proporción de Vectores de Soporte (VS) es la métrica de eficiencia:

* **Kernel Lineal:** Presenta un desempeño subóptimo (Exactitud ~0.70). La alta proporción de vectores de soporte (98-100%) indica un subajuste (*underfitting*), el modelo es incapaz de separar la topología circular de los datos, forzando a casi todas las muestras a ser vectores de soporte.
  `<img src="images/Linear_C1.0.png" width="300"/>`
* **Kernels Polinomiales (Homogéneo e Inhomogéneo):** Logran una clasificación perfecta (Exactitud 1.0) con la menor proporción de vectores de soporte (~2.6% - 7.3%). Esto confirma que, dado que nuestros datos son anillos concéntricos, el mapeo polinomial de grado 2 encuentra una estructura matemática intrínseca (una elipse) que separa las clases con máxima parsimonia.
  `<img src="images/Poly_Homo_C1.0.png" width="300"/>`
  `<img src="images/Poly_Inhomo_C1.0.png" width="300"/>`
* **Kernel RBG (Gaussiano):** Logra exactitud perfecta pero con una proporción de vectores de soporte mucho mayor (12% - 71%). Esto demuestra que el RBF, al ser extremadamente flexible (dimensión infinita), requiere más puntos de anclaje para "dibujar" la frontera, siendo más sensible al ruido y a la configuración de sus hiperparámetros.
  `<img src="images/RBF_C1.0.png" width="300"/>`
* **Kernel Sigmoide:** Presenta resultados variables. Debido a que no siempre satisface la Condición de Mercer (ser definido positivo), el optimizador en ocasiones encuentra dificultades topológicas, resultando en un modelo menos estable que los kernels polinomiales para este conjunto de datos específico.
  `<img src="images/Sigmoid_C1.0.png" width="300"/>`

| Modelo / Kernel               | Exactitud | Puntuación F1 | Vectores de Soporte | Proporción VS (%) | Tiempo (s) |
| ----------------------------- | --------- | -------------- | ------------------- | ------------------ | ---------- |
| Lineal C=1.0                  | 0.7067    | 0.3603         | 147                 | 98.00              | 5.0265     |
| Polinomial Inhomogéneo C=1.0 | 1.000     | 1.000          | 5                   | 3.33               | 2.7491     |
| Polinomial Homogéneo C=1.0   | 1.000     | 1.000          | 4                   | 2.67               | 2.3428     |
| RBF C=1.0                     | 1.000     | 1.000          | 38                  | 25.33              | 5.2613     |
| Sigmoide C=1.0                | 0.500     | 0.500          | 150                 | 100.00             | 0.3222     |

### Impacto de la Regularización ``C``

El parámetro ``C`` controla el equilibrio entre el margen y la clasificación correcta:

* Con ``C = 0.1``, el modelo es más "tolerante" (Margen Suave más amplio), requiriendo más vectores en algunos casos.
* Con ``C = 10.0``, el modelo penaliza más los errores, forzando una frontera más ajustada.

## 5. Conclusión

Se ha demostrado que la elección del Kernel es el factor determinante en el rendimiento de una SVM. Para distribuciones con estructuras geométricas claras como nuestros anillos concéntricos, el Kernel Polinomial de grado 2 ofrece el equilibrio perfecto entre precisión predictiva y eficiencia computacional (baja cantidad de vectores de soporte). Por el contrario, kernels más complejos o no definidos positivos (como el Sigmoide) pueden introducir inestabilidad en la optimización sin ofrecer ventajas significativas sobre el modelo polinomial.

---

⌨️ made with ❤️ by [Gustavo Rivero](https://github.com/gustavoerivero)
