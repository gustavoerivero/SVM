import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VISUALS_DIR = os.path.join(BASE_DIR, "images")

RNG = None

def setup() -> None:
    """
    Inicializa el entorno, crea los directorios necesarios y establece la semilla estocástica global.
    """
    global RNG
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)
    RNG = np.random.default_rng(42)

def get_separable_data(num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Genera un conjunto de datos bidimensional linealmente separable para clasificación binaria.
    """
    if RNG is None: raise RuntimeError("El entorno no ha sido inicializado. Ejecute setup().")
    X1 = RNG.standard_normal((num_samples, 2)) + np.array([2.5, 2.5])
    X2 = RNG.standard_normal((num_samples, 2)) + np.array([-2.5, -2.5])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
    return X, y

def get_overlapping_data(num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un conjunto de datos bidimensional no linealmente separable con clústeres superpuestos.
    """
    if RNG is None: raise RuntimeError("El entorno no ha sido inicializado. Ejecute setup().")
    X1 = RNG.standard_normal((num_samples, 2)) + np.array([0.5, 0.5])
    X2 = RNG.standard_normal((num_samples, 2)) + np.array([-0.5, -0.5])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
    return X, y

def get_tangible_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga, estandariza y reduce dimensionalmente el conjunto de datos empírico Breast Cancer Wisconsin.
    
    Se emplea StandardScaler para centrar los datos (μ=0, σ=1), un requisito matemático indispensable 
    para la convergencia de la SVM. Posteriormente, PCA proyecta las 30 dimensiones originales 
    a un espacio bidimensional (R²) para permitir su inspección visual y geométrica.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Componentes principales (X) y etiquetas en formato {-1, 1} (y).
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

def fit_dual_soft_margin(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    Ajusta una SVM de Margen Suave utilizando la Formulación Dual.
    
    Se precomputa la matriz del Kernel Lineal (K = X @ X.T) y la matriz Hessiana para
    maximizar (minimizando su negativo) la función objetivo. Luego se reconstruyen los pesos
    (w) y el sesgo (b) utilizando exclusivamente los Vectores de Soporte descubiertos.
    
    Objetivo a minimizar: Obj(α) = 0.5 * αᵀ H α - 1ᵀ α
    Sujeto a: ∑ αᵢyᵢ = 0
              0 ≤ αᵢ ≤ C
    """
    num_samples, _ = X.shape
    
    K = np.dot(X, X.T)
    H = np.outer(y, y) * K

    def dual_objective(alphas: np.ndarray) -> float:
        return 0.5 * np.dot(alphas, np.dot(H, alphas)) - np.sum(alphas)
    
    def dual_gradient(alphas: np.ndarray) -> np.ndarray:
        return np.dot(H, alphas) - np.ones(num_samples)

    constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y}
    bounds = [(0.0, C) for _ in range(num_samples)]
    initial_alphas = np.zeros(num_samples)

    result = opt.minimize(
        dual_objective,
        initial_alphas,
        method='SLSQP',
        jac=dual_gradient,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )

    if result.success:
        alphas = result.x
        
        sv_indices = alphas > 1e-5
        
        w_optimal = np.sum((alphas[sv_indices] * y[sv_indices])[:, None] * X[sv_indices], axis=0)
        
        free_sv = (alphas > 1e-5) & (alphas < C - 1e-5)
        if np.any(free_sv):
            idx = np.where(free_sv)[0][0]
        else:
            idx = np.where(sv_indices)[0][0] 
            
        b_optimal = y[idx] - np.dot(w_optimal, X[idx])
        
        return w_optimal, b_optimal, alphas
    else:
        print(f"Optimización fallida: {result.message}")
        return None, None, None
    
def export_results(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], b: Optional[float], alphas: Optional[np.ndarray], base_filename: str) -> None:
    """
    Crea un DataFrame de pandas, muestra una vista previa en consola y exporta los datos 
    a formato CSV y Excel, incluyendo el cálculo de los Multiplicadores de Lagrange (α).
    """
    data_dict = {
        'Característica_1': X[:, 0],
        'Característica_2': X[:, 1],
        'Etiqueta_Real': y
    }

    if w is not None and b is not None and alphas is not None:
        decision_values = np.dot(X, w) + b
        predictions = np.sign(decision_values)
        predictions[predictions == 0] = 1

        data_dict['Valor_Decisión'] = decision_values
        data_dict['Multiplicador_Alpha'] = alphas
        data_dict['Es_Vector_Soporte'] = alphas > 1e-5
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

def plot_svm(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], b: Optional[float], alphas: Optional[np.ndarray], title_message: str, image_filename: str) -> None:
    """
    Visualiza el conjunto de datos en el plano bidimensional, traza el hiperplano óptimo y 
    resalta explícitamente mediante contornos dorados aquellos puntos que son Vectores de Soporte (α > 0).
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='navy', marker='o', alpha=0.7, label='Clase Positiva (+1)')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='maroon', marker='x', alpha=0.7, label='Clase Negativa (-1)')
    
    if w is not None and b is not None:
        x_axis = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 100)
        y_axis = -(w[0] * x_axis + b) / w[1]
        
        margin = 1.0 / np.linalg.norm(w)
        margin_y = margin * np.sqrt(1 + (w[0]/w[1])**2)
        
        plt.plot(x_axis, y_axis, 'k-', linewidth=1.5, label='Hiperplano Óptimo')
        plt.plot(x_axis, y_axis + margin_y, 'k--', linewidth=1, alpha=0.6, label='Margen Superior')
        plt.plot(x_axis, y_axis - margin_y, 'k--', linewidth=1, alpha=0.6, label='Margen Inferior')
        
        if alphas is not None:
            sv_indices = alphas > 1e-5
            if np.any(sv_indices):
                plt.scatter(X[sv_indices][:, 0], X[sv_indices][:, 1], s=120, facecolors='none', edgecolors='gold', linewidths=2, label='Vectores de Soporte (α > 0)')
        
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
    setup()
    C_param = 1.0

    print("\n=== EVALUACIÓN: CONJUNTO LINEALMENTE SEPARABLE (FORMULACIÓN DUAL) ===")
    X_sep, y_sep = get_separable_data()
    w_sep, b_sep, a_sep = fit_dual_soft_margin(X_sep, y_sep, C=C_param)

    if w_sep is not None:
        print("\nOptimización Dual convergente.")
        export_results(X_sep, y_sep, w_sep, b_sep, a_sep, "datos_separables_dual")
        plot_svm(X_sep, y_sep, w_sep, b_sep, a_sep, f"SVM Margen Suave (Dual, C={C_param}) - Separables", "grafico_separable_dual.png")
    
    print("\n=== EVALUACIÓN: CONJUNTO CON CLASES SUPERPUESTAS (FORMULACIÓN DUAL) ===")
    X_mix, y_mix = get_overlapping_data()
    w_mix, b_mix, a_mix = fit_dual_soft_margin(X_mix, y_mix, C=C_param)
    
    if w_mix is not None:
        print("\nConvergencia exitosa. Vectores de soporte identificados en clústeres superpuestos.")
        export_results(X_mix, y_mix, w_mix, b_mix, a_mix, "datos_superpuestos_dual")
        plot_svm(X_mix, y_mix, w_mix, b_mix, a_mix, f"SVM Margen Suave (Dual, C={C_param}) - Superpuestas", "grafico_superpuesto_dual.png")

    print("\n=== EVALUACIÓN EMPÍRICA: BREAST CANCER WISCONSIN (PCA 2D) ===")
    X_real, y_real = get_tangible_data()
    
    w_real, b_real, a_real = fit_dual_soft_margin(X_real, y_real, C=C_param)
    
    if w_real is not None:
        print("\nConvergencia exitosa en datos empíricos de alta dimensionalidad.")
        export_results(X_real, y_real, w_real, b_real, a_real, "datos_reales_breast_cancer_dual")
        plot_svm(X_real, y_real, w_real, b_real, a_real, f"SVM Margen Suave Dual (Datos Reales PCA, C={C_param})", "grafico_real_breast_cancer_dual.png")

if __name__ == '__main__':
    main()