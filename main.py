import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt
import time

class AnomalyDetector:
    """
    Nienadzorowany detektor anomalii oparty na globalnych i lokalnych wskaźnikach niepodobieństwa do sąsiadów
    """
    
    def __init__(self, n_neighbors=5, contamination=0.172, metric='euclidean', method='combined'):
        """
        Inicjalizacja detektora
        
        :param n_neighbors: liczba sąsiadów do analizy
        :param contamination: szacowany procent anomalii w danych (%)
        :param metric: metryka odległości (może być dowolną metryką obsługiwaną przez sklearn.neighbors)
        :param method: metoda wykrywania anomalii ('global', 'local' lub 'combined')
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.method = method
        self.scaler = StandardScaler()
        
    def fit(self, X):
        """
        Dopasowanie modelu do danych
        """
        # Normalizacja danych
        X = self.scaler.fit_transform(X)
        self.X_ = X
        
        # Znajdź k najbliższych sąsiadów dla każdego punktu
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1, metric=self.metric).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Oblicz globalne i lokalne wskaźniki niepodobieństwa
        self.global_scores_ = np.mean(distances[:, 1:], axis=1)  # pomijamy samego siebie (indeks 0)
        self.local_scores_ = distances[:, self.n_neighbors]  # odległość do k-tego sąsiada
        
        # Połącz wyniki w zależności od wybranej metody
        if self.method == 'global':
            self.scores_ = self.global_scores_
        elif self.method == 'local':
            self.scores_ = self.local_scores_
        else:  # combined
            self.scores_ = self.global_scores_ * self.local_scores_
            
        # Oblicz próg dla zadanej frakcji anomalii
        self.threshold_ = np.percentile(self.scores_, 100 * (1 - self.contamination))
        
        return self
    
    def predict(self, X):
        """
        Predykcja etykiet (1 = normalny, -1 = anomalia)
        """
        X = self.scaler.transform(X)
        
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(self.X_)
        distances, _ = nbrs.kneighbors(X)
        
        global_scores = np.mean(distances, axis=1)
        local_scores = distances[:, -1]
        
        if self.method == 'global':
            scores = global_scores
        elif self.method == 'local':
            scores = local_scores
        else:  # combined
            scores = global_scores * local_scores
            
        return np.where(scores > self.threshold_, -1, 1)
    
    def decision_function(self, X):
        """
        Funkcja decyzyjna - zwraca wyniki anomalii (im wyższa wartość, tym większa szansa na bycie anomalią)
        """
        X = self.scaler.transform(X)
        
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(self.X_)
        distances, _ = nbrs.kneighbors(X)
        
        global_scores = np.mean(distances, axis=1)
        local_scores = distances[:, -1]
        
        if self.method == 'global':
            scores = global_scores
        elif self.method == 'local':
            scores = local_scores
        else:  # combined
            scores = global_scores * local_scores
            
        return scores
    
def load_datasets():
    """Ładowanie stabilnych zbiorów danych do eksperymentów"""
    from sklearn.datasets import fetch_openml
    import pandas as pd
    import numpy as np
    
    datasets = {}
    
    # 1. Zbiór danych Credit Card Fraud Detection (zmniejszony)
    try:
        credit = fetch_openml(name='creditcard', version=1, as_frame=True)
        X_credit = credit.data
        y_credit = credit.target.astype(int)
        # Losujemy próbkę, aby przyspieszyć obliczenia
        sample_idx = np.random.RandomState(42).choice(len(X_credit), 10000, replace=False)
        datasets['Credit Card'] = (X_credit.iloc[sample_idx], y_credit.iloc[sample_idx])
    except:
        print("Nie udało się załadować zbioru Credit Card, pomijam...")
    
    if not datasets:
        raise ValueError("Nie udało się załadować żadnego zbioru danych. Sprawdź połączenie internetowe.")
    
    return datasets
def evaluate_models(datasets, metrics=['euclidean', 'cosine']):
    """Ewaluacja różnych metod detekcji anomalii z obsługą przypadków skrajnych"""
    results = []
    
    for name, (X, y) in datasets.items():
        print(f"\n--- Ewaluacja na zbiorze {name} ---")
        print(f"Rozkład klas: {pd.Series(y).value_counts()}")
        
        # Usuń brakujące wartości i zduplikowane kolumny
        X = X.dropna(axis=1)
        X = X.loc[:, ~X.columns.duplicated()]
        
        # Sprawdź, czy mamy wystarczającą liczbę anomalii
        if len(np.unique(y)) < 2:
            print(f"Ostrzeżenie: Zbiór {name} zawiera tylko jedną klasę. Pomijam...")
            continue
            
        # Parametry do przetestowania
        param_grid = {
            'n_neighbors': [5, 10, 20],
            'contamination': [0.05, 0.1, 0.2],
            'metric': metrics,
            'method': ['global', 'local', 'combined']
        }
        
        # Testuj różne kombinacje parametrów
        best_score = 0
        best_params = None
        
        for params in ParameterGrid(param_grid):
            try:
                detector = AnomalyDetector(**params)
                detector.fit(X)
                pred_scores = detector.decision_function(X)
                
                # Sprawdź, czy mamy różne klasy w predykcjach
                if len(np.unique(y)) >= 2:
                    score = roc_auc_score(y, pred_scores)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        
                    results.append({
                        'dataset': name,
                        'method': 'Neighbor-based',
                        'params': str(params),
                        'auc': score
                    })
            except Exception as e:
                print(f"Błąd dla parametrów {params}: {str(e)}")
                continue
                
        if best_params is not None:
            print(f"Najlepszy wynik dla {name}: AUC = {best_score:.4f}, parametry: {best_params}")
        else:
            print(f"Nie udało się uzyskać wyników dla zbioru {name}")
        
        # Porównanie z innymi algorytmami (tylko jeśli mamy różne klasy)
        if len(np.unique(y)) >= 2:
            # Isolation Forest
            for n_estimators in [50, 100, 200]:
                for contamination in [0.05, 0.1, 0.2]:
                    try:
                        clf = IsolationForest(n_estimators=n_estimators, 
                                            contamination=contamination, 
                                            random_state=42)
                        clf.fit(X)
                        pred_scores = -clf.decision_function(X)
                        score = roc_auc_score(y, pred_scores)
                        
                        results.append({
                            'dataset': name,
                            'method': 'Isolation Forest',
                            'params': f"n_estimators={n_estimators}, contamination={contamination}",
                            'auc': score
                        })
                    except Exception as e:
                        print(f"Błąd w Isolation Forest: {str(e)}")
                        continue
            
            # One-Class SVM
            for kernel in ['rbf', 'linear']:
                for nu in [0.05, 0.1, 0.2]:
                    try:
                        clf = OneClassSVM(kernel=kernel, nu=nu)
                        clf.fit(X)
                        pred_scores = -clf.decision_function(X)
                        score = roc_auc_score(y, pred_scores)
                        
                        results.append({
                            'dataset': name,
                            'method': 'One-Class SVM',
                            'params': f"kernel={kernel}, nu={nu}",
                            'auc': score
                        })
                    except Exception as e:
                        print(f"Błąd w One-Class SVM: {str(e)}")
                        continue
    
    return pd.DataFrame(results)

def analyze_results(results_df):
    """Analiza wyników eksperymentów"""
    # Najlepsze wyniki dla każdego zbioru danych i metody
    best_results = results_df.loc[results_df.groupby(['dataset', 'method'])['auc'].idxmax()]
    
    print("\n--- Najlepsze wyniki dla każdej metody ---")
    print(best_results[['dataset', 'method', 'params', 'auc']].sort_values(['dataset', 'auc'], ascending=[True, False]))
    
    # Wizualizacja
    plt.figure(figsize=(12, 6))
    for dataset in results_df['dataset'].unique():
        subset = best_results[best_results['dataset'] == dataset]
        plt.bar(subset['method'] + ' - ' + dataset, subset['auc'], label=dataset)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('AUC')
    plt.title('Porównanie metod detekcji anomalii')
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.show()

def run_experiment():
    """Główna funkcja uruchamiająca eksperyment"""
    # Ładowanie danych
    datasets = load_datasets()
    
    # Ewaluacja modeli
    start_time = time.time()
    results_df = evaluate_models(datasets)
    print(f"\nCzas wykonania eksperymentów: {time.time() - start_time:.2f} sekund")
    
    # Analiza wyników
    analyze_results(results_df)
    
    return results_df

# Uruchomienie eksperymentu
if __name__ == "__main__":
    results = run_experiment()