import pandas as pd


class AnomalyDetectionModel():
    """
    Base class for all Anomaly Detection models used in project.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        """
        Abstract method for fitting the models to given training data.
        """
        raise NotImplementedError(f'[{self.__class__.__name__}] has no' +
                                  f'[{self.fit.__name__} implemented.]')

    def predict(self, X: pd.DataFrame):
        """
        Abstract method for predicting y value on given X input.
        """
        raise NotImplementedError(f'[{self.__class__.__name__}] has no' +
                                  f'[{self.fit.__name__} implemented.]')
