import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from models.AnomalyDetectionModel import AnomalyDetectionModel

class NeighborBasedAnomalyDetector(AnomalyDetectionModel):
    """
    Unsupervised anomaly detector based on global and local neighbor dissimilarity measures.
    Inherits from the base AnomalyDetectionModel class.
    """

    def __init__(self, n_neighbors: int = 5, contamination: float = 0.172, 
                 metric: str = 'euclidean', method: str = 'combined'):
        """
        Initialize the detector
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors to analyze (default=5)
        contamination : float
            Estimated percentage of anomalies in the data (0-1, default=0.172)
        metric : str
            Distance metric (any metric supported by sklearn.neighbors, default='euclidean')
        method : str
            Anomaly detection method ('global', 'local', or 'combined', default='combined')
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.method = method
        self.scaler = StandardScaler()
        self.X_ = None
        self.threshold_ = None

    def fit(self, X: pd.DataFrame, Y: pd.Series = None) -> None:
        """
        Fit the model to the training data (unsupervised - Y is ignored)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data (features)
        Y : pd.Series, optional
            Target values (ignored in unsupervised learning)
        """
        # Convert to numpy array if input is DataFrame
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        # Normalize data
        X_scaled = self.scaler.fit_transform(X_values)
        self.X_ = X_scaled
        
        # Find k nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1, 
                               metric=self.metric).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        
        # Calculate global and local dissimilarity scores
        self.global_scores_ = np.mean(distances[:, 1:], axis=1)  # skip self (index 0)
        self.local_scores_ = distances[:, self.n_neighbors]  # distance to k-th neighbor
        
        # Combine scores based on selected method
        if self.method == 'global':
            self.scores_ = self.global_scores_
        elif self.method == 'local':
            self.scores_ = self.local_scores_
        else:  # combined
            self.scores_ = self.global_scores_ * self.local_scores_
            
        # Calculate threshold for the given contamination rate
        self.threshold_ = np.percentile(self.scores_, 100 * (1 - self.contamination))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels (1 = normal, -1 = anomaly)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data to predict
            
        Returns:
        --------
        np.ndarray
            Array of predicted labels (-1 for anomalies, 1 for normal)
        """
        # Convert to numpy array if input is DataFrame
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        X_scaled = self.scaler.transform(X_values)
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, 
                               metric=self.metric).fit(self.X_)
        distances, _ = nbrs.kneighbors(X_scaled)
        
        global_scores = np.mean(distances, axis=1)
        local_scores = distances[:, -1]
        
        if self.method == 'global':
            scores = global_scores
        elif self.method == 'local':
            scores = local_scores
        else:  # combined
            scores = global_scores * local_scores
            
        return np.where(scores > self.threshold_, -1, 1)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute anomaly scores (higher values = more anomalous)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data to score
            
        Returns:
        --------
        np.ndarray
            Array of anomaly scores
        """
        # Convert to numpy array if input is DataFrame
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        
        X_scaled = self.scaler.transform(X_values)
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, 
                               metric=self.metric).fit(self.X_)
        distances, _ = nbrs.kneighbors(X_scaled)
        
        global_scores = np.mean(distances, axis=1)
        local_scores = distances[:, -1]
        
        if self.method == 'global':
            return global_scores
        elif self.method == 'local':
            return local_scores
        return global_scores * local_scores

    def fit_predict(self, X: pd.DataFrame, Y: pd.Series = None) -> np.ndarray:
        """
        Fit the model and return predictions for the training data
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data
        Y : pd.Series, optional
            Target values (ignored)
            
        Returns:
        --------
        np.ndarray
            Predicted labels
        """
        self.fit(X, Y)
        return self.predict(X)