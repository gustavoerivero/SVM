# SVM de Margen Suave (Soft Margin) - Formulación Dual

---

Este repositorio expone la culminación teórica del modelo SVM lineal mediante la implementación de la **Formulación Dual**. Al transicionar del problema primal al dual mediante los Multiplicadores de Lagrange ($\alpha$), el algoritmo revela que la frontera de decisión depende exclusivamente de un subconjunto crítico de los datos: los **Vectores de Soporte**.

## 1. Estructura del código

Este proyecto está organizado para facilitar la trazabilidad de los experimentos.

* ``src/``: Contiene el código fuente principal ([main.py](./src/main.py "Código fuente")).
* ``data/``: Almacena los resultados tabulados en formato CSV, incluyendo los multiplicadores de Lagrange.
* ``images/``: Contiene la visualización de los hiperplanos y los vectores de soporte identificados.

## 2. Guía de Ejecución

### Requisitos Previos

Asegúrese de tener instalado el entorno de Python con las bibliotecas necesarias:

```bash
pip install numpy pandas scipy matplotlib
```

### Ejecución

Para evaluar ambos escenarios (datos separables y superpuestos), ejecute:

```bash
cd src
python main.py
```

## 3. Descripción del Código

El sistema demuestra la elegancia de optimizar en el espacio de los multiplicadores de Lagrange en lugar del espacio de características original.

### Formulación Matemática

* *Función Objetivo (a Maximizar):*

  $\max_{\alpha} \sum_{i} \alpha_i - \frac{1}{2} \sum_{i} \sum_{j} \alpha_i \alpha_j y_i y_j (x_i^T x_j)$

* *Restricciones:*
  
  $\sum_{i} \alpha_i y_i = 0$
  $0 \leq \alpha_i \leq C, \quad \forall i$

Donde:

* $\alpha_i$ son los Multiplicadores de Lagrange.
* $C$ es la penalización de margen suave.
* $x_i^T x_j$ representa el producto punto (Kernel lineal) entre las muestras.

### Funciones Principales

* `fit_dual_soft_margin()`: Construye la matriz Hessiana y ajusta el modelo minimizando la forma cuadrática del problema dual con `SLSQP`. Reconstruye los pesos $w$ y el sesgo $b$ a partir de los $\alpha$ óptimos.
* `plot_svm()`: Visualiza la frontera y resalta, mediante anillos dorados, exclusivamente a los Vectores de Soporte ($\alpha_i > 0$).

## 4. Análisis de Resultados

El experimento demuestra la robustez del Margen Suave frente al colapso del Margen Rígido.

### Escenarios de Prueba

| Escenario          | Resultado              | Interpretación                                                                  |
| ------------------ | ---------------------- | -------------------------------------------------------------------------------- |
| Datos separables   | Convergencia exitosa   | Solo los puntos estrictamente en los bordes del margen resultan tener $\alpha > 0$.                 |
| Datos superpuestos | Convergencia exitosa | Los puntos dentro del margen (infracciones) alcanzan la restricción superior de caja ($\alpha = C$). |

<div style="display: flex; flex-direction: row;">
  <img src="./images/grafico_separable_primal.png" width="600" alt="Datos Separables" />
  <img src="./images/grafico_superpuesto_primal.png" width="600" alt="Datos Superpuestos" />
</div>

* **Interpretación del fallo:** La formulación dual prueba empíricamente la propiedad de parsimonia (esparsidad) de la SVM. Como se observa en los gráficos, la inmensa mayoría de los puntos de entrenamiento reciben un $\alpha = 0$, siendo completamente ignorados para el cálculo final del hiperplano. Únicamente los puntos resaltados en dorado dictan la posición geométrica de la frontera.

## 5. Conclusión

Resolver el problema SVM a través de su modelo Dual no es un mero ejercicio algebraico, sino la llave maestra del aprendizaje estadístico avanzado. Al demostrar que la solución depende estrictamente del producto punto de los vectores de entrada ($x_i^T x_j$), esta implementación sienta las bases matemáticas irrefutables para la introducción posterior de Funciones Kernel, permitiendo proyectar datos hacia dimensiones infinitas sin incrementar la complejidad computacional.

---

⌨️ made with ❤️ by [Gustavo Rivero](https://github.com/gustavoerivero)
