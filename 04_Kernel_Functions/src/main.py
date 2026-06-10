import os
import time
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable, Dict, Any, List

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

def kernel_linear(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Calcula el Kernel Lineal."""
    return np.dot(X1, X2.T)

def kernel_polynomial(X1: np.ndarray, X2: np.ndarray, degree: int = 3) -> np.ndarray:
    """Calcula el Kernel Polinomial Homogéneo."""
    return np.dot(X1, X2.T) ** degree

def kernel_rbf(X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Calcula el Kernel Gaussiano (Radial Basis Function)."""
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

def kernel_sigmoid(X1: np.ndarray, X2: np.ndarray, gamma: float = 0.01, coef0: float = 0.0) -> np.ndarray:
    """Calcula el Kernel Sigmoidal."""
    return np.tanh(gamma * np.dot(X1, X2.T) + coef0)

def get_nonlinear_data(num_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un conjunto de datos bidimensional no linealmente separable (topología circular).
    """
    if RNG is None: raise RuntimeError("El entorno no ha sido inicializado. Ejecute setup().")
    radius_inner = RNG.uniform(0, 1.5, num_samples)
    angle_inner = RNG.uniform(0, 2 * np.pi, num_samples)
    X1 = np.c_[radius_inner * np.cos(angle_inner), radius_inner * np.sin(angle_inner)]
    
    radius_outer = RNG.uniform(2.5, 4.0, num_samples)
    angle_outer = RNG.uniform(0, 2 * np.pi, num_samples)
    X2 = np.c_[radius_outer * np.cos(angle_outer), radius_outer * np.sin(angle_outer)]
    
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

def decision_function(X_test: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, alphas: np.ndarray, b: float, kernel_func: Callable, kernel_params: Dict[str, Any]) -> np.ndarray:
    """
    Evalúa la frontera de decisión utilizando la sumatoria de los núcleos de los Vectores de Soporte.
    """
    K = kernel_func(X_test, X_train, **kernel_params)
    return np.dot(K, alphas * y_train) + b

def fit_kernel_svm(X: np.ndarray, y: np.ndarray, C: float, kernel_func: Callable, kernel_params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Ajuste del modelo SVM Dual inyectando dinámicamente la función Kernel.
    """
    num_samples = X.shape[0]
    K = kernel_func(X, X, **kernel_params)
    H = np.outer(y, y) * K

    def dual_objective(alphas: np.ndarray) -> float:
        return 0.5 * float(np.dot(alphas, np.dot(H, alphas))) - float(np.sum(alphas))
    
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
        
        free_sv = (alphas > 1e-5) & (alphas < C - 1e-5)
        if np.any(free_sv):
            idx = np.where(free_sv)[0][0]
        else:
            idx = np.where(sv_indices)[0][0] 
            
        b_optimal = y[idx] - np.sum(alphas * y * K[:, idx])
        
        return alphas, b_optimal
    else:
        return None, None

def export_results(X: np.ndarray, y: np.ndarray, decision_values: Optional[np.ndarray], alphas: Optional[np.ndarray], base_filename: str) -> None:
    """
    Exporta los resultados tabulados de la clasificación a formato CSV y Excel.
    """
    data_dict = {
        'Característica_1': X[:, 0],
        'Característica_2': X[:, 1],
        'Etiqueta_Real': y
    }

    if decision_values is not None and alphas is not None:
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

    try:
        excel_filename = os.path.join(DATA_DIR, f"{base_filename}.xlsx")
        df.to_excel(excel_filename, index=False)
    except ImportError:
        pass

def export_benchmark_table(metrics_list: List[Dict[str, Any]], dataset_name: str) -> None:
    """
    Imprime en consola el cuadro comparativo de rendimiento y lo exporta como un reporte final.
    """
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.set_index('Núcleo (Kernel)', inplace=True)
    
    print(f"\n--- CUADRO COMPARATIVO DE RENDIMIENTO: {dataset_name.upper()} ---")
    print(df_metrics.to_string())
    print("-" * 75)
    
    csv_filename = os.path.join(DATA_DIR, f"comparativa_benchmark_{dataset_name}.csv")
    df_metrics.to_csv(csv_filename)
    
    try:
        excel_filename = os.path.join(DATA_DIR, f"comparativa_benchmark_{dataset_name}.xlsx")
        df_metrics.to_excel(excel_filename)
        print(f"Reporte de benchmark exportado exitosamente a Excel en '{excel_filename}'.")
    except ImportError:
        pass

def plot_kernel_svm(X: np.ndarray, y: np.ndarray, alphas: np.ndarray, b: float, kernel_func: Callable, kernel_params: Dict[str, Any], title_message: str, image_filename: str) -> None:
    """
    Visualiza el conjunto de datos y traza las fronteras de decisión no lineales mediante contornos.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='navy', marker='o', alpha=0.7, label='Clase Positiva (+1)')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='maroon', marker='x', alpha=0.7, label='Clase Negativa (-1)')
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = decision_function(grid_points, X, y, alphas, b, kernel_func, kernel_params)
    Z = Z.reshape(xx.shape)
    
    plt.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1], alpha=0.6)
    
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
    plt.close()

def plot_3d_mapping(X: np.ndarray, y: np.ndarray, image_filename: str) -> None:
    """
    Proyecta datos 2D a R³ mediante una transformación polinomial para visualizar el hiperplano real.
    """
    Z_dim = X[:, 0]**2 + X[:, 1]**2
    X_3d = np.c_[X, Z_dim]
    
    clf = opt.minimize(
        lambda w: 0.5 * float(np.dot(w[:-1], w[:-1])),
        np.zeros(4),
        constraints={'type': 'ineq', 'fun': lambda w: y * (np.dot(X_3d, w[:-1]) + w[-1]) - 1.0}
    )
    w_3d, b_3d = clf.x[:-1], clf.x[-1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[y == 1][:, 0], X_3d[y == 1][:, 1], X_3d[y == 1][:, 2], color='navy', marker='o', alpha=0.8, label='Clase +1')
    ax.scatter(X_3d[y == -1][:, 0], X_3d[y == -1][:, 1], X_3d[y == -1][:, 2], color='maroon', marker='x', alpha=0.8, label='Clase -1')
    
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 20), np.linspace(X[:, 1].min(), X[:, 1].max(), 20))
    zz = -(w_3d[0] * xx + w_3d[1] * yy + b_3d) / (w_3d[2] + 1e-10)
    
    ax.plot_surface(xx, yy, zz, color='gold', alpha=0.3)
    ax.set_title("Proyección a R³: Separabilidad Lineal en Alta Dimensionalidad", fontsize=12)
    
    save_path = os.path.join(VISUALS_DIR, image_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_kernels(X: np.ndarray, y: np.ndarray, dataset_name: str) -> None:
    """
    Itera sobre los núcleos, calcula métricas de rendimiento y genera artefactos visuales y tabulares.
    """
    C_param = 1.0
    num_total_samples = len(y)
    metrics_list = []
    
    kernels = {
        "Lineal": (kernel_linear, {}),
        "Polinomial": (kernel_polynomial, {'degree': 2}),
        "RBF_Gaussiano": (kernel_rbf, {'gamma': 1.0}),
        "Sigmoidal": (kernel_sigmoid, {'gamma': 0.05, 'coef0': 0.0})
    }

    print(f"\nIniciando procesamiento y benchmarking para el conjunto: {dataset_name}...")

    for name, (k_func, k_params) in kernels.items():
        start_time = time.time()
        alphas, b = fit_kernel_svm(X, y, C_param, k_func, k_params)
        elapsed_time = time.time() - start_time
        
        if alphas is not None:
            dec_vals = decision_function(X, X, y, alphas, b, k_func, k_params)
            
            predictions = np.sign(dec_vals)
            predictions[predictions == 0] = 1
            accuracy = float(np.mean(predictions == y) * 100)
            
            num_sv = int(np.sum(alphas > 1e-5))
            
            metrics_list.append({
                'Núcleo (Kernel)': name,
                'Tiempo (Segundos)': round(elapsed_time, 4),
                'Vectores de Soporte': f"{num_sv} / {num_total_samples}",
                'Tasa de VS (%)': round((num_sv / num_total_samples) * 100, 2),
                'Precisión (%)': round(accuracy, 2)
            })
            
            export_results(X, y, dec_vals, alphas, f"datos_{dataset_name}_{name.lower()}")
            plot_kernel_svm(X, y, alphas, b, k_func, k_params, f"SVM Margen Suave (Kernel {name}) - {dataset_name}", f"grafico_{dataset_name}_{name.lower()}.png")
        else:
            print(f"Fallo crítico en la optimización convexa para el Kernel {name}.")
            metrics_list.append({
                'Núcleo (Kernel)': name,
                'Tiempo (Segundos)': round(elapsed_time, 4),
                'Vectores de Soporte': "N/A",
                'Tasa de VS (%)': "N/A",
                'Precisión (%)': "N/A (Fallo)"
            })
            
    export_benchmark_table(metrics_list, dataset_name)

def main() -> None:
    setup()

    print("\n" + "="*60)
    print("EVALUACIÓN 1: DATOS SINTÉTICOS NO LINEALES (CIRCULARES)")
    print("="*60)
    X_circ, y_circ = get_nonlinear_data(num_samples=50) 
    evaluate_kernels(X_circ, y_circ, "sinteticos")
    
    print("\n" + "="*60)
    print("EVALUACIÓN 2: DATOS EMPÍRICOS BREAST CANCER WISCONSIN (PCA 2D)")
    print("="*60)
    X_real, y_real = get_tangible_data()
    evaluate_kernels(X_real, y_real, "breast_cancer")

    print("\n=== GENERANDO DEMOSTRACIÓN DEL TEOREMA DE MERCER EN R³ ===")
    plot_3d_mapping(X_circ, y_circ, "grafico_proyeccion_3d.png")
    print("Demostración completada exitosamente.")

if __name__ == '__main__':
    main()