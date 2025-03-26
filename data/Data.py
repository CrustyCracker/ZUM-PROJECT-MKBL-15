from copy import deepcopy
import pandas as pd
import os
from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_DATASETS = [name for name in os.listdir(CURRENT_DIR)
                    if os.path.isdir(os.path.join(CURRENT_DIR, name))
                    and not name.startswith("__")]


class CreditCardData:
    '''
    Data class for credit card fraud detection dataset.
        self.raw_data: the raw data
        self.preprocessed_data: the preprocessed data
        self.preprocessed_x: the preprocessed data without the class attribute
        self.preprocessed_y: the class attribute
    '''

    def __init__(self, dataset='creditcard', distinct_columns=None) -> None:
        """
        Initialize data class with given dataset and preprocess it.
        """
        if dataset not in ALLOWED_DATASETS:
            raise ValueError(f"Given set [{dataset}] not listed. Please,"
                  + f" choose one from the following:\n {ALLOWED_DATASETS}")
        self.dataset = dataset

        raw_data = pd.read_csv(os.path.join(CURRENT_DIR,
                               os.path.join(dataset, "creditcard.csv")))
        self.raw_data = raw_data
        self._preprocess(distinct_columns)

    def get_train_and_valid_set(self, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split preprocessed data into training and validation sets.
        Stratified sampling to maintain class imbalance.
        """
        from sklearn.model_selection import train_test_split
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            self.preprocessed_x,
            self.preprocessed_y,
            test_size=1-frac,
            stratify=self.preprocessed_y,
            random_state=42
        )
        
        train = pd.concat([X_train, y_train], axis=1)
        valid = pd.concat([X_valid, y_valid], axis=1)
        
        return train, valid

    def _preprocess(self, distinct_columns) -> None:
        """
        Preprocess credit card transaction data.
        """
        raw_data = deepcopy(self.raw_data)
        
        # 1. Handle Time feature - convert to hours of day
        raw_data['Time'] = raw_data['Time'] % (24 * 3600) / 3600
        
        # 2. Scale Amount and Time features (V1-V28 are already PCA components)
        scaler = RobustScaler()
        raw_data[['Time', 'Amount']] = scaler.fit_transform(raw_data[['Time', 'Amount']])
        
        # 3. Optionally add derived features
        raw_data['Amount_Time_Ratio'] = raw_data['Amount'] / (raw_data['Time'] + 1)
        
        # 4. Handle class imbalance (just separate, no resampling here)
        if distinct_columns:
            if not all([col in raw_data.columns for col in distinct_columns]):
                raise ValueError("There are columns which are not present in" +
                               "preprocessed data.")
            raw_data = raw_data[distinct_columns + ['Class']]
        
        self.preprocessed_data = raw_data
        self.preprocessed_x = raw_data.drop('Class', axis=1)
        self.preprocessed_y = raw_data['Class']

    def get_feature_names(self) -> list:
        """Return list of feature names"""
        return list(self.preprocessed_x.columns)


def preprocess_creditcard(raw_data: pd.DataFrame, distinct_columns=None) -> pd.DataFrame:
    """
    Preprocessing function for credit card fraud data
    """
    # Make copy to avoid modifying original
    data = raw_data.copy()
    
    # 1. Convert Time to cyclical features
    data['Time_sin'] = np.sin(2 * np.pi * data['Time']/max(data['Time']))
    data['Time_cos'] = np.cos(2 * np.pi * data['Time']/max(data['Time']))
    data = data.drop('Time', axis=1)
    
    # 2. Scale Amount using RobustScaler
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    
    # 3. Optionally select specific columns
    if distinct_columns:
        if not all([col in data.columns for col in distinct_columns]):
            raise ValueError("Specified columns not found in data")
        data = data[distinct_columns + ['Class']]
    
    return data


PREPROCESS_FUNCTIONS = {
    "creditcard": preprocess_creditcard,
    # Other datasets can be added here
}