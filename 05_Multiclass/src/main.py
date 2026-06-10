import os
import time
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable, Dict, Any, List

from sklearn.datasets import make_blobs, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VISUALS_DIR = os.path.join(BASE_DIR, "images")

RNG = None

def setup() -> None:
    """Inicializa el entorno y establece la semilla global."""
    global RNG
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)
    RNG = np.random.default_rng(42)

# --- NÚCLEOS MATEMÁTICOS ---
def kernel_linear(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """Calcula el Kernel Lineal."""
    return np.dot(X1, X2.T)

def kernel_rbf(X1: np.ndarray, X2: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Calcula el Kernel Gaussiano (RBF)."""
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

# --- GENERADORES DE DATOS (5 CLASES) ---
def get_synthetic_multiclass(num_samples: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """Genera 5 clústeres sintéticos en R² con leve superposición."""
    if RNG is None: setup()
    X, y = make_blobs(n_samples=num_samples, centers=5, cluster_std=1.2, random_state=42)
    return X, y

def get_empirical_multiclass() -> Tuple[np.ndarray, np.ndarray]:
    """Carga dígitos empíricos (0 al 4), estandariza y aplica PCA a R²."""
    data = load_digits(n_class=5)
    X_raw, y = data.data, data.target
    
    indices = []
    for c in range(5):
        idx_c = np.where(y == c)[0][:30]
        indices.extend(idx_c)
    
    X_raw = X_raw[indices]
    y = y[indices]
    
    X_scaled = StandardScaler().fit_transform(X_raw)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    
    return X_pca, y

# --- NÚCLEO SVM BINARIO BASE ---
def fit_binary_svm(X: np.ndarray, y: np.ndarray, C: float, kernel_func: Callable, kernel_params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Ajuste matemático del modelo binario inyectando dinámicamente la función Kernel."""
    num_samples = X.shape[0]
    K = kernel_func(X, X, **kernel_params)
    H = np.outer(y, y) * K

    def dual_objective(alphas: np.ndarray) -> float:
        return 0.5 * float(np.dot(alphas, np.dot(H, alphas))) - float(np.sum(alphas))
    
    def dual_gradient(alphas: np.ndarray) -> np.ndarray:
        return np.dot(H, alphas) - np.ones(num_samples)

    bounds = [(0.0, C) for _ in range(num_samples)]
    constraints = {'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y}
    
    result = opt.minimize(dual_objective, np.zeros(num_samples), method='SLSQP', jac=dual_gradient, bounds=bounds, constraints=constraints, options={'maxiter': 500})

    if result.success:
        alphas = result.x
        sv_idx = alphas > 1e-5
        free_sv = (alphas > 1e-5) & (alphas < C - 1e-5)
        idx = np.where(free_sv)[0][0] if np.any(free_sv) else (np.where(sv_idx)[0][0] if np.any(sv_idx) else 0)
        b = y[idx] - np.sum(alphas * y * K[:, idx])
        return alphas, b
    return None, None

def decision_function(X_test: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, alphas: np.ndarray, b: float, kernel_func: Callable, kernel_params: Dict[str, Any]) -> np.ndarray:
    """Evalúa la frontera de decisión utilizando la sumatoria de los núcleos de los Vectores de Soporte."""
    K = kernel_func(X_test, X_train, **kernel_params)
    return np.dot(K, alphas * y_train) + b

# --- ESTRATEGIAS MULTICLASE ---
class MultiClassSVM:
    def __init__(self, strategy: str, kernel: Callable, kernel_params: dict, C: float = 1.0):
        self.strategy = strategy
        self.kernel = kernel
        self.k_params = kernel_params
        self.C = C
        self.models = []
        self.classes = []
        self.X_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        self.X_train = X
        
        if self.strategy == 'OvR':
            for c in self.classes:
                y_bin = np.where(y == c, 1, -1)
                alphas, b = fit_binary_svm(X, y_bin, self.C, self.kernel, self.k_params)
                if alphas is not None: self.models.append((alphas, b, y_bin, c))
                
        elif self.strategy == 'OvO':
            for i in range(len(self.classes)):
                for j in range(i + 1, len(self.classes)):
                    c1, c2 = self.classes[i], self.classes[j]
                    mask = (y == c1) | (y == c2)
                    X_pair, y_pair = X[mask], y[mask]
                    y_bin = np.where(y_pair == c1, 1, -1)
                    
                    alphas, b = fit_binary_svm(X_pair, y_bin, self.C, self.kernel, self.k_params)
                    if alphas is not None: self.models.append((alphas, b, X_pair, y_bin, c1, c2))

    def predict_detailed(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Devuelve las predicciones y la matriz de justificación (márgenes o votos)."""
        if self.strategy == 'OvR':
            decisions = np.zeros((X_test.shape[0], len(self.classes)))
            for i, (alphas, b, y_bin, c) in enumerate(self.models):
                decisions[:, i] = decision_function(X_test, self.X_train, y_bin, alphas, b, self.kernel, self.k_params)
            predictions = self.classes[np.argmax(decisions, axis=1)]
            return predictions, decisions
            
        elif self.strategy == 'OvO':
            votes = np.zeros((X_test.shape[0], len(self.classes)))
            for alphas, b, X_tr_pair, y_bin, c1, c2 in self.models:
                decs = decision_function(X_test, X_tr_pair, y_bin, alphas, b, self.kernel, self.k_params)
                preds = np.where(decs > 0, c1, c2)
                for k, p in enumerate(preds):
                    votes[k, np.where(self.classes == p)[0][0]] += 1
            predictions = self.classes[np.argmax(votes, axis=1)]
            return predictions, votes

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        preds, _ = self.predict_detailed(X_test)
        return preds

# --- VISUALIZACIÓN Y REPORTES ---
def plot_multiclass_boundaries(X: np.ndarray, y: np.ndarray, model: MultiClassSVM, title_message: str, image_filename: str):
    """Mapea el área 2D coloreando las zonas de decisión sin resaltar Vectores de Soporte."""
    plt.figure(figsize=(9, 7))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # Relleno de las regiones de decisión multiclase
    plt.contourf(xx, yy, Z, alpha=0.35, cmap='tab10')
    
    # Ploteo exclusivo de las clases (sin los anillos dorados de los VS)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='tab10', s=45, linewidth=0.8)
    
    # Leyenda para las 5 clases
    legend_labels = [f"Clase {i}" for i in range(5)]
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels, title="Categorías", loc='best')
    
    plt.title(title_message, fontsize=12, pad=15)
    plt.xlabel('Característica 1', fontsize=11)
    plt.ylabel('Característica 2', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(VISUALS_DIR, image_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def export_detailed_predictions(X: np.ndarray, y_real: np.ndarray, y_pred: np.ndarray, justification_matrix: np.ndarray, strat: str, k_name: str, dataset_name: str):
    """Exporta el desglose analítico de cada inferencia (trazabilidad)."""
    data = {
        'Característica_1': X[:, 0],
        'Característica_2': X[:, 1],
        'Etiqueta_Real': y_real,
        'Predicción_SVM': y_pred,
        'Clasificación_Correcta': y_real == y_pred
    }
    
    metric_name = "Margen_Modelo_Clase" if strat == 'OvR' else "Votos_Clase"
    for i in range(justification_matrix.shape[1]):
        data[f'{metric_name}_{i}'] = justification_matrix[:, i]
        
    df = pd.DataFrame(data)
    csv_filename = os.path.join(DATA_DIR, f"trazabilidad_{dataset_name}_{strat.lower()}_{k_name.lower()}.csv")
    df.to_csv(csv_filename, index=False)

def run_multiclass_benchmarks(X: np.ndarray, y: np.ndarray, dataset_name: str):
    """Ejecuta y compara las estrategias OvR y OvO."""
    print(f"\nIniciando procesamiento y benchmarking multiclase para el conjunto: {dataset_name}...")
    
    configs = [
        ('OvR', 'Lineal', kernel_linear, {}),
        ('OvR', 'RBF_Gaussiano', kernel_rbf, {'gamma': 1.0}),
        ('OvO', 'Lineal', kernel_linear, {}),
        ('OvO', 'RBF_Gaussiano', kernel_rbf, {'gamma': 1.0})
    ]
    
    metrics = []
    for strat, k_name, k_func, k_params in configs:
        print(f"Entrenando estrategia: {strat} con Kernel {k_name}...")
        t0 = time.time()
        
        clf = MultiClassSVM(strategy=strat, kernel=k_func, kernel_params=k_params, C=1.0)
        clf.fit(X, y)
        
        t_elapsed = time.time() - t0
        
        preds, justification = clf.predict_detailed(X)
        accuracy = float(np.mean(preds == y) * 100)
        n_models = len(clf.models)
        
        export_detailed_predictions(X, y, preds, justification, strat, k_name, dataset_name)
        
        metrics.append({
            'Estrategia': strat,
            'Núcleo (Kernel)': k_name,
            'Modelos Entrenados': n_models,
            'Tiempo (Segundos)': round(t_elapsed, 4),
            'Precisión (%)': round(accuracy, 2)
        })
        
        plot_name = f"grafico_multiclase_{dataset_name}_{strat.lower()}_{k_name.lower()}.png"
        title_msg = f"Estrategia {strat} | Kernel {k_name} | Conjunto: {dataset_name.capitalize()}"
        plot_multiclass_boundaries(X, y, clf, title_msg, plot_name)

    df_metrics = pd.DataFrame(metrics).set_index(['Estrategia', 'Núcleo (Kernel)'])
    
    print(f"\n--- CUADRO COMPARATIVO DE RENDIMIENTO MULTICLASE: {dataset_name.upper()} ---")
    print(df_metrics.to_string())
    print("-" * 80)
    
    csv_path = os.path.join(DATA_DIR, f"comparativa_benchmark_multiclase_{dataset_name}.csv")
    df_metrics.to_csv(csv_path)
    
    try: 
        excel_path = os.path.join(DATA_DIR, f"comparativa_benchmark_multiclase_{dataset_name}.xlsx")
        df_metrics.to_excel(excel_path)
        print(f"Reporte de benchmark exportado exitosamente a Excel en '{excel_path}'.")
    except ImportError: 
        pass

def main():
    setup()
    
    print("\n" + "="*70)
    print("EVALUACIÓN 1: DATOS SINTÉTICOS MULTICLASE (5 CLÚSTERES)")
    print("="*70)
    X_syn, y_syn = get_synthetic_multiclass()
    run_multiclass_benchmarks(X_syn, y_syn, "sinteticos")
    
    print("\n" + "="*70)
    print("EVALUACIÓN 2: DATOS EMPÍRICOS DIGITS (PCA 2D)")
    print("="*70)
    X_emp, y_emp = get_empirical_multiclass()
    run_multiclass_benchmarks(X_emp, y_emp, "empiricos")
    
    print("\nBenchmark multiclase completado. Artefactos exportados en los directorios correspondientes.")

if __name__ == '__main__':
    main()