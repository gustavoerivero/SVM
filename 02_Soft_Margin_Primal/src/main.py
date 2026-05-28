import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VISUALS_DIR = os.path.join(BASE_DIR, "images")

# Generador estocástico centralizado
RNG = None

def setup_dirs() -> None:
    """
    Crea los directorios necesarios para almacenar los datos y los artefactos visuales.
    """
    global RNG
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)
    RNG = np.random.default_rng(42)

def get_separable_data(num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]: 
    """Genera un conjunto de datos bidimensional linealmente separable para clasificación binaria."""
    if RNG is None: raise RuntimeError("El entorno no ha sido inicializado. Ejecute setup_dirs().")
    
    X1 = RNG.standard_normal((num_samples, 2)) + np.array([2.5, 2.5])
    X2 = RNG.standard_normal((num_samples, 2)) + np.array([-2.5, -2.5])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
    return X, y

def get_overlapping_data(num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Genera un conjunto de datos bidimensional no linealmente separable con clústeres superpuestos."""
    if RNG is None: raise RuntimeError("El entorno no ha sido inicializado. Ejecute setup_dirs().")
    
    X1 = RNG.standard_normal((num_samples, 2)) + np.array([0.5, 0.5])
    X2 = RNG.standard_normal((num_samples, 2)) + np.array([-0.5, -0.5])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
    return X, y

def get_tangible_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga, estandariza y reduce dimensionalmente el conjunto de datos empírico Breast Cancer Wisconsin.
    """
    data = load_breast_cancer()
    X_raw = data.data
    y_raw = data.target
    
    y = np.where(y_raw == 0, -1, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, y

def fit_primal_soft_margin(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    Ajusta una Máquina de Vectores de Soporte de Margen Suave utilizando la formulación primal.
    """
    num_samples, num_features = X.shape

    def objective(params: np.ndarray) -> float:
        w = params[:num_features]
        xi = params[num_features + 1:]
        return 0.5 * float(np.dot(w, w)) + C * float(np.sum(xi))
    
    def constraint(params: np.ndarray) -> np.ndarray:
        w = params[:num_features]
        b = params[num_features]
        xi = params[num_features + 1:]
        return y * (np.dot(X, w) + b) - 1.0 + xi

    initial_guess = np.zeros(num_features + 1 + num_samples)
    
    bounds_list = [(None, None)] * (num_features + 1) + [(0.0, None)] * num_samples
    constraints = {'type': 'ineq', 'fun': constraint}

    result = minimize(
        objective,
        initial_guess,
        method='SLSQP',
        bounds=bounds_list,
        constraints=constraints,
        options={'maxiter': 1500}
    )

    if result.success:
        w_optimal = result.x[:num_features]
        b_optimal = float(result.x[num_features])
        xi_optimal = result.x[num_features + 1:]
        return w_optimal, b_optimal, xi_optimal
    else:
        print(f"Optimización fallida: {result.message}")
        return None, None, None
    
def export_results(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], b: Optional[float], xi: Optional[np.ndarray], base_filename: str) -> None:
    """
    Crea un DataFrame de pandas, muestra una vista previa en consola y exporta los datos a CSV y Excel.
    """
    data_dict = {
        'Característica_1': X[:, 0],
        'Característica_2': X[:, 1],
        'Etiqueta_Real': y
    }

    if w is not None and b is not None and xi is not None:
        decision_values = np.dot(X, w) + b
        predictions = np.sign(decision_values)
        predictions[predictions == 0] = 1

        data_dict['Valor_Decisión'] = decision_values
        data_dict['Variable_Holgura_Xi'] = xi
        data_dict['Invasión_de_Margen'] = xi > 1e-4
        data_dict['Predicción_SVM'] = predictions
        data_dict['Clasificación_Correcta'] = (predictions == y)

    df = pd.DataFrame(data_dict)

    print(f"\n--- Vista Previa de los Datos: {base_filename} ---")
    print(df.head(5).to_string())
    print("...")

    csv_filename = os.path.join(DATA_DIR, f"{base_filename}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Éxito: Datos exportados a '{csv_filename}'")

    try:
        excel_filename = os.path.join(DATA_DIR, f"{base_filename}.xlsx")
        df.to_excel(excel_filename, index=False)
        print(f"Éxito: Datos exportados a '{excel_filename}'")
    except ImportError:
        print("Aviso: No se pudo exportar a Excel por falta de la biblioteca 'openpyxl'.")

def plot_svm(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], b: Optional[float], xi: Optional[np.ndarray], title_message: str, image_filename: str) -> None:
    """
    Visualiza el conjunto de datos y resalta aquellos puntos que invaden el margen (ξ > 0).
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
        
        if xi is not None:
            slack_indices = xi > 1e-4
            if np.any(slack_indices):
                plt.scatter(X[slack_indices][:, 0], X[slack_indices][:, 1], s=120, facecolors='none', edgecolors='gold', linewidths=2, label='Infracciones de Margen (ξ > 0)')
        
    plt.title(title_message, fontsize=12, pad=15)
    plt.xlabel('Característica 1', fontsize=11)
    plt.ylabel('Característica 2', fontsize=11)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(VISUALS_DIR, image_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Éxito: Gráfico exportado e inmortalizado en '{save_path}'")
    plt.close()

def main() -> None:
    setup_dirs()
    
    C_param = 1.0

    print("\n=== EVALUACIÓN: CONJUNTO LINEALMENTE SEPARABLE (PRIMAL SOFT MARGIN) ===")
    X_sep, y_sep = get_separable_data()
    w_sep, b_sep, xi_sep = fit_primal_soft_margin(X_sep, y_sep, C=C_param)

    if w_sep is not None:
        print("\nOptimización convergente en sintonía perfecta.")
        export_results(X_sep, y_sep, w_sep, b_sep, xi_sep, "datos_separables_primal")
        plot_svm(X_sep, y_sep, w_sep, b_sep, xi_sep, f"SVM Margen Suave (Primal, C={C_param}) - Separables", "grafico_separable_primal.png")
    else:
        print("\nFallo inesperado en la optimización del conjunto separable.")

    print("\n=== EVALUACIÓN: CONJUNTO CON CLASES SUPERPUESTAS (PRIMAL SOFT MARGIN) ===")
    X_mix, y_mix = get_overlapping_data()
    w_mix, b_mix, xi_mix = fit_primal_soft_margin(X_mix, y_mix, C=C_param)
    
    if w_mix is not None:
        print("\nConvergencia exitosa lograda gracias a las variables de holgura (ξ).")
        export_results(X_mix, y_mix, w_mix, b_mix, xi_mix, "datos_superpuestos_primal")
        plot_svm(X_mix, y_mix, w_mix, b_mix, xi_mix, f"SVM Margen Suave (Primal, C={C_param}) - Superpuestas", "grafico_superpuesto_primal.png")
    else:
        print("\nEl modelo colapsó inesperadamente.")

    print("\n=== EVALUACIÓN EMPÍRICA: BREAST CANCER WISCONSIN (PCA 2D) ===")
    X_real, y_real = get_tangible_data()
    
    w_real, b_real, xi_real = fit_primal_soft_margin(X_real, y_real, C=C_param)
    
    if w_real is not None:
        print("\nConvergencia exitosa en datos empíricos de alta dimensionalidad.")
        export_results(X_real, y_real, w_real, b_real, xi_real, "datos_reales_breast_cancer_primal")
        plot_svm(X_real, y_real, w_real, b_real, xi_real, f"SVM Margen Suave Primal (Datos Reales PCA, C={C_param})", "grafico_real_breast_cancer_primal.png")
    else:
        print("\nEl modelo colapsó inesperadamente en el conjunto empírico.")

if __name__ == '__main__':
    main()