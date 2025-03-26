from models.AnomalyDetectionModel import AnomalyDetectionModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ModelEvaluator():
    '''
    Base class for evaluating model's metrics.
      self._confusion_matrix: the confusion matrix for the model
                              [[tp, fp], [fn, tn]]
      self._accuracy:(tp + tn) / (tp + tn + fp + fn)
      self._recall: (also known as sensitivity) tp / (tp + fn)
      self._sensitivity: (also known as recall) tp / (tp + fn)
      self._specificity: tn / (tn + fp)
      self._precision: tp / (tp + fp)
      self._f1_score: 2 * (precision * recall) / (precision + recall)
      self._support: (tp + fn)
    '''

    def __init__(self, model: AnomalyDetectionModel) -> None:
        self.model = model

        self._confusion_matrix = None
        self._accuracy = None
        self._recall = None
        self._sensitivity = None
        self._specificity = None
        self._precision = None
        self._f1_score = None
        self._support = None

        self._fpr = None
        self._tpr = None
        self._auc = None

    def calculate_quality(self, input: pd.DataFrame, y_true: np.array,
                          cutoff=0.5) -> None:
        '''
        Calculates quality metrics for the model.
          input:  dataset on which the quality will be calculated
          y_true: list of true values
          y_pred: list of PREDICTED PROBABILITIES
        '''

        y_pred = self.model.predict(input)

        self._calculate_confusion_matrix(y_true, y_pred, cutoff)
        self._calculate_accuracy()
        self._calculate_recall()
        self._calculate_sensitivity()
        self._calculate_specificity()
        self._calculate_precision()
        self._calculate_f1_score()
        self._calculate_support()

    def plot_roc_curve(self, axs: plt.Axes, input: pd.DataFrame,
                       y_true: np.array, cutoffs: list) -> None:
        '''
        Plots the roc curve for the model.
        '''
        y_pred = self.model.predict(input)

        self._calculate_roc_curve(y_true, y_pred, cutoffs)
        if self._fpr is None or self._tpr is None:
            raise ValueError(
                "ROC curve has not been calculated yet. \
                 Use calculate_roc_curve() first.")
        axs.plot(self._fpr, self._tpr)
        axs.plot([0, 1], [0, 1], linestyle='--')
        axs.set_xlabel('False Positive Rate')
        axs.set_ylabel('True Positive Rate')
        axs.set_title(f"{self.model.name} AUC: {self._auc:.3f}")

    def plot_confusion_matrix(self, axs: plt.Axes) -> None:
        '''
        Plots the confusion matrix for the model.
        '''
        if self._confusion_matrix is None:
            raise ValueError(
                "Confusion matrix has not been calculated yet. \
                 Use calculate_quality() first.")

        plt.figure(figsize=(10, 10))
        g = sns.heatmap(self._confusion_matrix,
                        xticklabels=["Positive", "Negative"],
                        yticklabels=["Positive", "Negative"],
                        annot=True,
                        fmt="d",
                        linewidths=5,
                        square=True,
                        annot_kws={"fontsize": 20},
                        cmap="Blues",
                        cbar=True,
                        ax=axs)
        g.set_xlabel('Actual Class')
        g.set_ylabel('Predicted Class')
        g.set_title(self.model.name)

    def get_confusion_matrix(self) -> np.array:
        '''Returns the confusion matrix for the model'''
        if self._confusion_matrix is None:
            raise ValueError(
                "Confusion matrix has not been calculated yet. \
                 Use calculate_quality() first.")
        return self._confusion_matrix

    def get_accuracy(self) -> float:
        '''Returns the accuracy for the model'''
        if self._accuracy is None:
            raise ValueError(
                "Accuracy has not been or cannot be calculated \
                 (devision by zero). Use calculate_quality() first.")
        return self._accuracy

    def get_recall(self) -> float:
        '''Returns the recall for the model'''
        if self._recall is None:
            raise ValueError(
                "Recall has not been or cannot be calculated \
                 (devision by zero). Use calculate_quality() first.")
        return self._recall

    def get_sensitivity(self) -> float:
        '''Returns the sensitivity for the model'''
        if self._sensitivity is None:
            raise ValueError(
                "Sensitivity has not been or cannot be calculated \
                 (devision by zero). Use calculate_quality() first.")
        return self._sensitivity

    def get_specificity(self) -> float:
        '''Returns the specificity for the model'''
        if self._specificity is None:
            raise ValueError(
                "Specificity has not been or cannot be calculated \
                 (devision by zero). Use calculate_quality() first.")
        return self._specificity

    def get_precision(self) -> float:
        '''Returns the precision for the model'''
        if self._precision is None:
            raise ValueError(
                "Precision has not been or cannot be calculated \
                 (devision by zero). Use calculate_quality() first.")
        return self._precision

    def get_f1_score(self) -> float:
        '''Returns the f1 score for the model'''
        if self._f1_score is None:
            raise ValueError(
                "F1 score has not been or cannot be calculated \
                 (devision by zero). Use calculate_quality() first.")
        return self._f1_score

    def get_support(self) -> float:
        '''Returns the support for the model'''
        if self._support is None:
            raise ValueError(
                "Support has not been or cannot be calculated \
                 (devision by zero). Use calculate_quality() first.")
        return self._support

    def get_auc(self) -> float:
        '''Returns the auc for the model'''
        if self._auc is None:
            raise ValueError(
                "AUC has not been or cannot be calculated (devision by zero). \
                 Use calculate_quality() first.")
        return self._auc

    def _calculate_confusion_matrix(self, y_true, y_pred, cutoff) -> None:
        '''Calculates the confusion matrix for the model'''
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        # ypred comes as a list of probabilities, we need
        #  to convert it to a list of 0s and 1s
        y_pred = [1 if p >= cutoff else 0 for p in y_pred]

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == 1 and y_p == 1:
                tp += 1
            elif y_t == 0 and y_p == 1:
                fp += 1
            elif y_t == 1 and y_p == 0:
                fn += 1
            elif y_t == 0 and y_p == 0:
                tn += 1
        self._confusion_matrix = np.array([[tp, fp], [fn, tn]])
        self._tp = tp
        self._fp = fp
        self._fn = fn
        self._tn = tn

    def _calculate_accuracy(self) -> None:
        '''Calculates the accuracy for the model'''
        try:
            self._accuracy = (self._tp + self._tn) / \
                (self._tp + self._tn + self._fp + self._fn)
        except ZeroDivisionError:
            self._accuracy = None

    def _calculate_recall(self) -> None:
        '''Calculates the recall for the model'''
        try:
            self._recall = self._tp / (self._tp + self._fn)
        except ZeroDivisionError:
            self._recall = None

    def _calculate_sensitivity(self) -> None:
        '''Calculates the sensitivity for the model'''
        try:
            self._sensitivity = self._tp / (self._tp + self._fn)
        except ZeroDivisionError:
            self._sensitivity = None

    def _calculate_specificity(self) -> None:
        '''Calculates the specificity for the model'''
        try:
            self._specificity = self._tn / (self._tn + self._fp)
        except ZeroDivisionError:
            self._specificity = None

    def _calculate_precision(self) -> None:
        '''Calculates the precision for the model'''
        try:
            self._precision = self._tp / (self._tp + self._fp)
        except ZeroDivisionError:
            self._precision = "inf"

    def _calculate_f1_score(self) -> None:
        '''Calculates the f1 score for the model'''
        try:
            self._f1_score = 2 * (self._precision * self._recall) / \
                (self._precision + self._recall)
        except ZeroDivisionError:
            self._f1_score = "inf"
        except TypeError:
            self._f1_score = "inf"

    def _calculate_support(self) -> None:
        '''Calculates the support for the model'''
        self._support = self._tp + self._fn

    def _calculate_roc_curve(self, y_true, y_pred, cutoffs: list) -> None:
        '''Calculates the roc curve for the model'''
        save_confusion_matrix = np.copy(self._confusion_matrix)
        save_sensitivity = self._sensitivity
        save_specificity = self._specificity
        try:
            self._fpr = []
            self._tpr = []

            for cutoff in cutoffs:
                self._calculate_confusion_matrix(y_true, y_pred, cutoff)
                self._calculate_sensitivity()
                self._calculate_specificity()
                self._fpr.append(1 - self._specificity)
                self._tpr.append(self._sensitivity)
            self._calulate_auc()
        finally:
            self._confusion_matrix = save_confusion_matrix
            self._sensitivity = save_sensitivity
            self._specificity = save_specificity

    def _calulate_auc(self):
        '''Calculates the auc for the model using trapezoidal rule'''
        if self._fpr is None or self._tpr is None:
            raise ValueError(
                "ROC curve has not been calculated yet. \
                 Use calculate_roc_curve() first.")
        self._auc = 0
        for i in range(len(self._fpr) - 1):
            self._auc += (self._tpr[i] + self._tpr[i + 1]) * \
                (self._fpr[i] - self._fpr[i + 1]) / 2

    def __str__(self) -> str:
        string = f"Model: {self.model.name}\n"
        if self._confusion_matrix is not None:
            string += f"Confusion Matrix: \n{self.get_confusion_matrix()}\n"
            string += f"Accuracy: {self.get_accuracy()}\n"
            string += "Recall(also known as Sensitivity): " + \
                      f"{self.get_recall()}\n"
            string += f"Specificity: {self.get_specificity()}\n"
            string += f"Precision: {self.get_precision()}\n"
            string += f"F1 Score: {self.get_f1_score()}\n"
            string += f"Support: {self.get_support()}\n"
        if self._auc is not None:
            string += f"AUC: {self.get_auc()}\n"
        return string


if __name__ == "__main__":
    print("This is a module. Import it to use it.")
