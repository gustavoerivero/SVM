import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def setup_dirs() -> None:
    """
    Crea los directorios necesarios para almacenar los datos y los artefactos visuales.
    """
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../images", exist_ok=True)

def get_separable_data(num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Genera un conjunto de datos bidimensional linealmente separable para clasificación binaria.

    Args:
        num_samples (int): Número de muestras por clase.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Características del conjunto de datos (X) y etiquetas (y).
    """
    np.random.seed(42)
    X1 = np.random.randn(num_samples, 2) + np.array([2.5, 2.5])
    X2 = np.random.randn(num_samples, 2) + np.array([-2.5, -2.5])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
    return X, y

def get_overlapping_data(num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un conjunto de datos bidimensional no linealmente separable con clústeres superpuestos.

    Args:
        num_samples (int): Número de muestras por clase.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Características del conjunto de datos (X) y etiquetas (y).
    """
    np.random.seed(42)
    X1 = np.random.randn(num_samples, 2) + np.array([0.5, 0.5])
    X2 = np.random.randn(num_samples, 2) + np.array([-0.5, -0.5])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
    return X, y

def fit_svm(X: np.ndarray, y: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Ajusta una Máquina de Vectores de Soporte de Margen Rígido utilizando la formulación de optimización primal.

    Args:
        X (np.ndarray): Características del conjunto de entrenamiento.
        y (np.ndarray): Etiquetas de clase correspondientes.

    Returns:
        Tuple[Optional[np.ndarray], Optional[float]]: Pesos del modelo (w) y sesgo (b).
    """
    _, num_features = X.shape

    def objective(params: np.ndarray) -> float:
        """
        Calcula la función objetivo a minimizar: 1/2 * ||w||^2.
        """
        w = params[:-1]
        return 0.5 * float(np.dot(w, w))
    
    def constraint(params: np.ndarray) -> np.ndarray:
        """
        Calcula la restricción de desigualdad estricta: y_i * (w^T * x_i + b) - 1 >= 0.
        """
        w = params[:-1]
        b = params[-1]
        return y * (np.dot(X, w) + b) - 1.0

    initial_guess = np.zeros(num_features + 1)
    constraints = {'type': 'ineq', 'fun': constraint}

    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if result.success:
        w_optimal = result.x[:-1]
        b_optimal = float(result.x[-1])
        return w_optimal, b_optimal
    else:
        print("Optimización fallida: ", result.message)
        return None, None
    
def export_results(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], b: Optional[float], base_filename: str) -> None:
    """
    Crea un DataFrame de pandas, muestra una vista previa en consola y exporta los datos a CSV y Excel.

    Args:
        X (np.ndarray): Características del conjunto de entrenamiento.
        y (np.ndarray): Etiquetas de clase correspondientes.
        w (Optional[np.ndarray]): Pesos del modelo (w).
        b (Optional[float]): Sesgo del modelo (b).
        base_filename (str): Nombre base para los archivos generados.
    """
    data_dict = {
        'Característica_1': X[:, 0],
        'Característica_2': X[:, 1],
        'Etiqueta_Real': y
    }

    if w is not None and b is not None:
        decision_values = np.dot(X, w) + b
        predictions = np.sign(decision_values)
        predictions[predictions == 0] = 1

        data_dict['Valor_Decisión'] = decision_values
        data_dict['Predicción_SVM'] = predictions
        data_dict['Clasificación_Correcta'] = (predictions == y)

    df = pd.DataFrame(data_dict)

    print(f"\n--- Vista Previa de los Datos: {base_filename} ---")
    print(df.head(5).to_string())
    print("...")

    csv_filename = f"../data/{base_filename}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Éxito: Datos exportados a '{csv_filename}'")

    try:
        excel_filename = f"../data/{base_filename}.xlsx"
        df.to_excel(excel_filename, index=False)
        print(f"Éxito: Datos exportados a '{excel_filename}'")
    except ImportError:
        print("Aviso: No se pudo exportar a Excel por falta de la biblioteca 'openpyxl'.")

def plot_svm(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], b: Optional[float], title_message: str, image_filename: str) -> None:
    """
    Visualiza el conjunto de datos, traza las fronteras óptimas si están disponibles y exporta la figura.

    Args:
        X (np.ndarray): Características del conjunto de entrenamiento.
        y (np.ndarray): Etiquetas de clase correspondientes.
        w (Optional[np.ndarray]): Pesos del modelo (w).
        b (Optional[float]): Sesgo del modelo (b).
        title_message (str): Título principal del gráfico.
        image_filename (str): Nombre del archivo para la exportación de la imagen.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='navy', marker='o', alpha=0.7, label='Clase Positiva (+1)')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='maroon', marker='x', alpha=0.7, label='Clase Negativa (-1)')
    
    if w is not None and b is not None:
        x_axis = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 100)
        y_axis = -(w[0] * x_axis + b) / w[1]
        
        margin = 1.0 / np.linalg.norm(w)
        margin_y = margin * np.sqrt(1 + (w[0]/w[1])**2)
        y_axis_up = y_axis + margin_y
        y_axis_down = y_axis - margin_y
        
        plt.plot(x_axis, y_axis, 'k-', linewidth=1.5, label='Hiperplano Óptimo')
        plt.plot(x_axis, y_axis_up, 'k--', linewidth=1, alpha=0.6, label='Margen Superior')
        plt.plot(x_axis, y_axis_down, 'k--', linewidth=1, alpha=0.6, label='Margen Inferior')
        
    plt.title(title_message, fontsize=12, pad=15)
    plt.xlabel('Característica 1', fontsize=11)
    plt.ylabel('Característica 2', fontsize=11)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = f"../images/{image_filename}"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Éxito: Gráfico exportado e inmortalizado en '{save_path}'")
    plt.close()

def main() -> None:
    setup_dirs()

    print("\n=== EVALUACIÓN: CONJUNTO LINEALMENTE SEPARABLE ===")
    X_sep, y_sep = get_separable_data()
    w_sep, b_sep = fit_svm(X_sep, y_sep)

    if w_sep is not None:
        print("\nOptimización convergente en sintonía perfecta.")
        export_results(X_sep, y_sep, w_sep, b_sep, "datos_separables")
        plot_svm(X_sep, y_sep, w_sep, b_sep, "SVM Margen Rígido - Clases Linealmente Separables", "grafico_separable.png")
    else:
        print("\nFallo inesperado en la optimización del conjunto separable.")

    print("\n=== EVALUACIÓN: CONJUNTO CON CLASES SUPERPUESTAS ===")
    X_mix, y_mix = get_overlapping_data()
    w_mix, b_mix = fit_svm(X_mix, y_mix)
    
    if w_mix is not None:
        print("\nConvergencia inesperada detectada.")
        export_results(X_mix, y_mix, w_mix, b_mix, "datos_superpuestos")
        plot_svm(X_mix, y_mix, w_mix, b_mix, "SVM Margen Rígido - Clases Superpuestas", "grafico_superpuesto.png")
    else:
        print("\nResultado esperado: El modelo de margen rígido colapsa por falta de separabilidad.")
        export_results(X_mix, y_mix, None, None, "datos_superpuestos")
        plot_svm(X_mix, y_mix, None, None, "Fallo de Optimización - Ausencia de Separabilidad Lineal", "grafico_superpuesto_fallo.png")

if __name__ == '__main__':
    main()