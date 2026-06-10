# Electiva II: Máquinas de Vectores de Soporte

**Maestría en Ciencias de la Computación**
**Universidad Centroccidental "Lisandro Alvarado"**
**Decanato de Ciencias y Tecnología**

---

## 🏛️ Descripción General

El siguiente repositorio documenta el desarrollo e implementación práctica de algoritmos de **Máquinas de Vectores de Soporte (SVM)**, realizados como parte de la asignatura _*Electiva II: Máquinas de Vectores de Soporte*_ bajo la guía del **Dr. Javier Hernández Benítez**.

## 📂 Estructura del Repositorio

| Proyecto            | Descripción                                                                                                            | Enlace                                                                               |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 01. Hard Margin SVM | Implementación del SVM bajo formulación primal con restricciones de margen estricto.                                   | [Ir al proyecto](https://github.com/gustavoerivero/SVM/tree/main/01_Hard_Margin_Primal) |
| 02. Soft Margin SVM | Implementación del SVM bajo formulación primal con restricciones de margen suave.                                      | [Ir al proyecto](https://github.com/gustavoerivero/SVM/tree/main/02_Soft_Margin_Primal) |
| 03. Soft Margin SVM - Modelo Dual | Implementación del SVM bajo formulación dual para la extracción analítica de Vectores de Soporte.      | [Ir al proyecto](https://github.com/gustavoerivero/SVM/tree/main/03_Dual_Model)  |
| 04. Funciones Kernel | Extensión del modelo Dual mediante el Teorema de Mercer para resolver topologías no lineales en alta dimensionalidad. | [Ir al proyecto](https://github.com/gustavoerivero/SVM/tree/main/04_Kernel_Functions) |
| 05. Estrategias Multiclase | Implementación y benchmarking computacional de las arquitecturas Uno-contra-Todos (OvR) y Uno-contra-Uno (OvO). | [Ir al proyecto](https://github.com/gustavoerivero/SVM/tree/main/05_Multiclass) |

## 🛠️ Stack Tecnológico

Para garantizar la reproducibilidad científica, el código ha sido desarrollado utilizando:

* **Python 3.x:** Lenguaje núcleo.
* **NumPy / Pandas:** Manipulación de tensores bidimensionales y generación de reportes analíticos.
* **SciPy (Optimize):** Resolución algorítmica de problemas de programación cuadrática (`SLSQP`).
* **Scikit-learn:** Extracción de conjuntos de datos empíricos (Oncológicos y Reconocimiento de Dígitos manuscritos) y reducción de dimensionalidad (PCA).
* **Matplotlib:** Renderizado de fronteras de decisión y visualizaciones geométricas de alta precisión.
* **Openpyxl:** Exportación de métricas matemáticas a formato Excel.

---

⌨️ made with ❤️ by [Gustavo Rivero](https://github.com/gustavoerivero)