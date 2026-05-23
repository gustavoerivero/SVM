import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, Optional, Callable, Dict, Any, List

# Resolución dinámica de rutas absolutas para garantizar estabilidad
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VISUALS_DIR = os.path.join(BASE_DIR, "images")

def setup_dirs() -> None:
    """
    Crea los directorios necesarios para almacenar los datos y los artefactos visuales.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)

def get_archetype_data(num_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera un conjunto de datos no linealmente separable simulando arquetipos de jugadores en anillos concéntricos.
    Args:
        num_samples (int): Número total de muestras a generar (debe ser par).
    Returns:
        Tuple[np.ndarray, np.ndarray]: Características del conjunto de datos (X) y etiquetas (y).
    """
    np.random.seed(42)
    angles = np.random.uniform(0, 2 * np.pi, num_samples)
    
    radius_inner = np.random.uniform(0, 1.5, num_samples // 2)
    x1_inner = radius_inner * np.cos(angles[:num_samples // 2])
    x2_inner = radius_inner * np.sin(angles[:num_samples // 2])
    y_inner = np.ones(num_samples // 2)
    
    radius_outer = np.random.uniform(2.5, 4.0, num_samples // 2)
    x1_outer = radius_outer * np.cos(angles[num_samples // 2:])
    x2_outer = radius_outer * np.sin(angles[num_samples // 2:])
    y_outer = -np.ones(num_samples // 2)
    
    X = np.vstack((np.column_stack((x1_inner, x2_inner)), np.column_stack((x1_outer, x2_outer))))
    y = np.hstack((y_inner, y_outer))
    
    noise = np.random.normal(0, 0.3, X.shape)
    return X + noise, y

def linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz del Kernel Lineal. Satisface el Teorema de Mercer.
    
    Función de Kernel: K(x, y) = xᵀy
    """
    return np.dot(X1, X2.T)

def poly_kernel(X1: np.ndarray, X2: np.ndarray, degree: int = 2, coef0: float = 1.0) -> np.ndarray:
    """
    Calcula la matriz del Kernel Polinomial Inhomogéneo. Satisface el Teorema de Mercer.
    
    Función de Kernel: K(x, y) = (xᵀy + coef0)ᵈ
    """
    return (np.dot(X1, X2.T) + coef0) ** degree

def poly_homogeneous_kernel(X1: np.ndarray, X2: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Calcula la matriz del Kernel Polinomial Homogéneo. Satisface el Teorema de Mercer.
    
    Función de Kernel: K(x, y) = (xᵀy)ᵈ
    """
    return (np.dot(X1, X2.T)) ** degree

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """
    Calcula la matriz del Kernel Gaussiano (RBF). Estrictamente definido positivo.
    
    Función de Kernel: K(x, y) = exp(-γ ||x - y||²)
    """
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)

def sigmoid_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float = 0.01, coef0: float = -0.1) -> np.ndarray:
    """
    Calcula la matriz del Kernel Sigmoide. 
    
    Función de Kernel: K(x, y) = tanh(γ(xᵀy) + θ)
    """
    return np.tanh(gamma * np.dot(X1, X2.T) + coef0)

class KernelSVM:
    """
    Máquina de Vectores de Soporte de Margen Suave (Soft Margin SVM).
    
    Función objetivo dual:
    Obj(α) = ½ * ∑ᵢ∑ⱼ αᵢαⱼyᵢyⱼK(xᵢ, xⱼ) - ∑ᵢ αᵢ
    """
    def __init__(self, kernel: Callable, C: float = 1.0) -> None:
        self.kernel = kernel
        self.C = C
        self.alphas: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Ajusta el modelo resolviendo el problema de optimización dual.

        Restricción de caja: 0 ≤ αᵢ ≤ C
        """
        num_samples = X.shape[0]
        K = self.kernel(X, X)
        
        self.X_train = X
        self.y_train = y
        
        def dual_obj(alphas: np.ndarray) -> float:
            # Obj(α) = ½ * ∑ᵢ∑ⱼ αᵢαⱼyᵢyⱼK(xᵢ, xⱼ) - ∑ᵢ αᵢ
            return 0.5 * np.sum(np.outer(alphas * y, alphas * y) * K) - np.sum(alphas)
            
        def dual_grad(alphas: np.ndarray) -> np.ndarray:
            # Gradiente: ∇Obj(α)ᵢ = ∑ⱼ αⱼyᵢyⱼK(xᵢ, xⱼ) - 1
            return np.dot(np.outer(y, y) * K, alphas) - np.ones(num_samples)

        bounds = Bounds(np.zeros(num_samples), np.full(num_samples, self.C))
        linear_constraint = LinearConstraint(y.reshape(1, -1), [0], [0])
        
        initial_alphas = np.zeros(num_samples)
        
        result = minimize(
            fun=dual_obj,
            x0=initial_alphas,
            jac=dual_grad,
            method='trust-constr',
            constraints=[linear_constraint],
            bounds=bounds,
            options={'maxiter': 1000, 'verbose': 0}
        )
        
        if result.success:
            self.alphas = result.x
            
            support_vector_indices = (self.alphas > 1e-5)
            if np.any(support_vector_indices):
                sv_idx = np.where((self.alphas > 1e-5) & (self.alphas < self.C - 1e-5))[0]
                if len(sv_idx) > 0:
                    idx = sv_idx[0]
                    # Cálculo del sesgo utilizando un vector de soporte que no esté en el límite de C
                    # Función de decisión para un vector de soporte: y_i * (∑ αⱼyⱼK(xⱼ, xᵢ) + b) = 1
                    self.b = y[idx] - np.sum(self.alphas * y * self.kernel(X, X[idx:idx+1].reshape(1, -1)).flatten())
                else:
                    idx = np.where(support_vector_indices)[0][0]
                    self.b = y[idx] - np.sum(self.alphas * y * self.kernel(X, X[idx:idx+1].reshape(1, -1)).flatten())
            return True
        return False

    def decision_func(self, X: np.ndarray) -> np.ndarray:
        """
        Evalúa la función de decisión f(x) = ∑ᵢ(αᵢyᵢK(xᵢ, x)) + b
        """
        if self.alphas is None or self.X_train is None or self.y_train is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir.")
        
        K = self.kernel(self.X_train, X)
        return np.dot((self.alphas * self.y_train), K) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clase: yₚ = sgn(f(x))"""
        return np.sign(self.decision_func(X))

def calc_metrics(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, sv_count: int, total_samples: int, exec_time: float) -> Dict[str, Any]:
    """
    Calcula métricas de clasificación exhaustivas para un modelo SVM entrenado.
    """
    sv_ratio = (sv_count / total_samples) * 100

    return {
        'Modelo / Kernel': model_name,
        'Exactitud': float(accuracy_score(y_true, y_pred)),
        'Precisión': float(precision_score(y_true, y_pred, zero_division=0)),
        'Exhaustividad': float(recall_score(y_true, y_pred, zero_division=0)),
        'Puntuación F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'Vectores de Soporte': int(sv_count),
        'Proporción VS (%)': float(sv_ratio),
        'Tiempo (seg)': float(exec_time)
    }

def print_report(metrics_list: List[Dict[str, Any]]) -> None:
    """
    Genera y formatea un reporte comparativo tabular para todos los modelos evaluados.
    """
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics.round(4)
    
    print("\n" + "="*95)
    print("REPORTE COMPARATIVO DE RENDIMIENTO DE KERNELS SVM (MARGEN SUAVE)".center(95))
    print("="*95)
    print(df_metrics.to_string(index=False))
    print("="*95 + "\n")
    
    csv_path = os.path.join(DATA_DIR, "reporte_comparativo_kernels.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Éxito: Reporte comparativo maestro exportado a '{csv_path}'")

def plot_svm(model: KernelSVM, X: np.ndarray, y: np.ndarray, title: str, filename: str) -> None:
    """
    Visualiza el conjunto de datos, la frontera de decisión no lineal y los márgenes mediante gráficos de contorno.
    """
    plt.figure(figsize=(9, 7))
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    Z = model.decision_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    z_min, z_max = Z.min(), Z.max()
    if np.isclose(z_min, z_max):
       z_min, z_max = -1.0, 1.0
    elif z_min >= 0:
        z_min = -0.1
    elif z_max <= 0:
        z_max = 0.1
    # --------------------------------------------------------------------------------
    
    plt.contourf(xx, yy, Z, levels=np.linspace(z_min, 0, 7), cmap='Reds', alpha=0.3)
    plt.contourf(xx, yy, Z, levels=np.linspace(0, z_max, 7), cmap='Blues', alpha=0.3)
    
    if not np.isclose(Z.min(), Z.max()):
        plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'], colors='k', linewidths=[1, 1.5, 1])
    
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='navy', marker='o', edgecolors='k', label='Arquetipo 1 (+1)')
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='maroon', marker='s', edgecolors='k', label='Arquetipo 2 (-1)')
    
    sv_indices = model.alphas > 1e-5
    if np.any(sv_indices):
        plt.scatter(X[sv_indices][:, 0], X[sv_indices][:, 1], s=100, facecolors='none', edgecolors='gold', linewidths=1.5, label='Vectores de Soporte')
    
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Dimensión Táctica', fontsize=11)
    plt.ylabel('Dimensión de Acción', fontsize=11)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(VISUALS_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Éxito: Gráfico de fronteras exportado a '{save_path}'")


def export_results(model: KernelSVM, X: np.ndarray, y: np.ndarray, filename: str) -> None:
    """
    Exporta los parámetros del modelo y las predicciones a un archivo CSV.
    """
    df = pd.DataFrame({
        'Dim_Táctica': X[:, 0],
        'Dim_Acción': X[:, 1],
        'Etiqueta_Real': y,
        'Valor_Decisión': model.decision_func(X),
        'Predicción': model.predict(X),
        'Multiplicador_Lagrange_Alpha': model.alphas,
        'Es_Vector_Soporte': (model.alphas > 1e-5)
    })
    
    csv_path = os.path.join(DATA_DIR, f"{filename}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Éxito: Datos detallados del modelo exportados a '{csv_path}'")

def main() -> None:
    """
    Tubería de ejecución para el entrenamiento con variación de parámetros C y múltiples Kernels.
    """
    setup_dirs()

    print("\n--- Generando Dataset de Arquetipos (Anillos Concéntricos) ---")
    X, y = get_archetype_data(num_samples=150)
    
    c_values = [0.1, 1.0, 10.0]
    
    kernels = {
        'Linear': linear_kernel,
        'Poly_Inhomo': lambda x1, x2: poly_kernel(x1, x2, degree=2),
        'Poly_Homo': lambda x1, x2: poly_homogeneous_kernel(x1, x2, degree=2),
        'RBF': lambda x1, x2: rbf_kernel(x1, x2, gamma=0.5),
        'Sigmoid': lambda x1, x2: sigmoid_kernel(x1, x2, gamma=0.01)
    }
    
    metrics_list = []
    
    for c_val in c_values:
        for name, kernel_func in kernels.items():
            model_id = f"{name}_C{c_val}"
            print(f"\n--- Entrenando Modelo: {model_id} ---")
            model = KernelSVM(kernel=kernel_func, C=c_val)
            
            start_time = time.time()
            success = model.fit(X, y)
            end_time = time.time()
            
            if success:
                exec_time = end_time - start_time
                y_pred = model.predict(X)
                sv_count = np.sum(model.alphas > 1e-5)
                
                metrics = calc_metrics(
                    model_name=model_id,
                    y_true=y,
                    y_pred=y_pred,
                    sv_count=sv_count,
                    total_samples=X.shape[0],
                    exec_time=exec_time
                )
                metrics_list.append(metrics)
                
                plot_svm(model, X, y, f"{model_id}", f"{model_id}.png")
                export_results(model, X, y, f"{model_id}")
            else:
                print(f"Advertencia: El optimizador falló en la convergencia para {model_id}.")
            
    if metrics_list:
        print_report(metrics_list)

if __name__ == '__main__':
    main()