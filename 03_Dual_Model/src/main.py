import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from typing import Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VISUALS_DIR = os.path.join(BASE_DIR, "images")

# Generador estocástico centralizado
RNG = None

def setup() -> None:
    """
    Inicializa el entorno y establece la semilla estocástica global.
    """
    global RNG
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)
    RNG = np.random.default_rng(42)

def get_separable_data(num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]: 
    """Genera un conjunto de datos bidimensional linealmente separable."""
    if RNG is None: raise RuntimeError("El entorno no ha sido inicializado. Ejecute setup().")
    X1 = RNG.standard_normal((num_samples, 2)) + np.array([2.5, 2.5])
    X2 = RNG.standard_normal((num_samples, 2)) + np.array([-2.5, -2.5])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
    return X, y

def get_overlapping_data(num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Genera un conjunto de datos bidimensional no linealmente separable."""
    if RNG is None: raise RuntimeError("El entorno no ha sido inicializado. Ejecute setup().")
    X1 = RNG.standard_normal((num_samples, 2)) + np.array([0.5, 0.5])
    X2 = RNG.standard_normal((num_samples, 2)) + np.array([-0.5, -0.5])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(num_samples), -np.ones(num_samples)))
    return X, y

def fit_dual_soft_margin(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """
    Ajusta una SVM utilizando la Formulación Dual.
    
    Maximiza (minimiza el negativo de): Obj(α) = ∑ αᵢ - ½ ∑ᵢ∑ⱼ αᵢαⱼyᵢyⱼ(xᵢᵀxⱼ)
    Sujeto a: ∑ αᵢyᵢ = 0
              0 ≤ αᵢ ≤ C
    """
    num_samples, _ = X.shape
    
    # Matriz del Kernel Lineal precomputada: K = X @ X.T
    K = np.dot(X, X.T)
    # Matriz Hessiana: H_ij = y_i * y_j * K_ij
    H = np.outer(y, y) * K

    def dual_objective(alphas: np.ndarray) -> float:
        # 0.5 * αᵀ H α - 1ᵀ α
        return 0.5 * np.dot(alphas, np.dot(H, alphas)) - np.sum(alphas)
    
    def dual_gradient(alphas: np.ndarray) -> np.ndarray:
        # H α - 1
        return np.dot(H, alphas) - np.ones(num_samples)

    # Restricción de igualdad: ∑ αᵢyᵢ = 0
    constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y}
    
    # Restricción de caja: 0 ≤ αᵢ ≤ C
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
        
        # Identificar Vectores de Soporte (α > 0)
        sv_indices = alphas > 1e-5
        
        # Reconstruir el vector de pesos (w = ∑ αᵢyᵢxᵢ)
        w_optimal = np.sum((alphas[sv_indices] * y[sv_indices])[:, None] * X[sv_indices], axis=0)
        
        # Calcular el sesgo (b) usando un Vector de Soporte Libre (0 < α < C)
        free_sv = (alphas > 1e-5) & (alphas < C - 1e-5)
        if np.any(free_sv):
            idx = np.where(free_sv)[0][0]
        else:
            idx = np.where(sv_indices)[0][0] # Fallback
            
        b_optimal = y[idx] - np.dot(w_optimal, X[idx])
        
        return w_optimal, b_optimal, alphas
    else:
        print(f"Optimización fallida: {result.message}")
        return None, None, None
    
def export_results(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], b: Optional[float], alphas: Optional[np.ndarray], base_filename: str) -> None:
    """Exporta los resultados y los Multiplicadores de Lagrange (α)."""
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
    csv_filename = os.path.join(DATA_DIR, f"{base_filename}.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Éxito: Datos exportados a '{csv_filename}'")

def plot_svm(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], b: Optional[float], alphas: Optional[np.ndarray], title_message: str, image_filename: str) -> None:
    """Visualiza el hiperplano y resalta los Vectores de Soporte (α > 0)."""
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

if __name__ == '__main__':
    main()