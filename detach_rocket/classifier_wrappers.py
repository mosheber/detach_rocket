from sklearn.base import ClassifierMixin
from sklearn.linear_model import (RidgeClassifierCV ,RidgeClassifier)
import numpy as np

class ClassifierWrapper:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def get_chosen_hyperparams(self) -> dict:
        pass

    def set_chosen_hyperparams(self, hyperparams: dict):
        pass

    def fit_new_model_on_chosen_hyperparams(self, X_train, y_train) -> ClassifierMixin:
        pass

    def get_full_classifier(self) -> ClassifierMixin:
        return self._full_classifier

class RidgeClassifierWrapper(ClassifierWrapper):
    def __init__(self, **kwargs):
        super().__init__()
        self._full_classifier = RidgeClassifierCV(alphas=np.logspace(-10,10,20))

    def fit(self, X, y):
        self._full_classifier.fit(X, y)

    def get_chosen_hyperparams(self) -> dict:
        if not hasattr(self, "_full_model_alpha"):
            self._full_model_alpha = self._full_classifier.alpha_

        print('Optimal Alpha Full ROCKET: {:.2f}'.format(self._full_model_alpha))
        return {
            "alpha": self._full_model_alpha
        }
    
    def set_chosen_hyperparams(self, hyperparams: dict):
        self._full_model_alpha = hyperparams["alpha"]

    def fit_new_model_on_chosen_hyperparams(self, X_train, y_train) -> ClassifierMixin:
        sfd_classifier = RidgeClassifier(alpha=self._full_model_alpha)
        sfd_classifier.fit(X_train, y_train)
        return sfd_classifier

    
CLASSIFIER_WRAPPER_MAP = {
    "RidgeClassifier": RidgeClassifierWrapper
}

def create_new_classifier(name, **kwargs):
    return CLASSIFIER_WRAPPER_MAP[name](**kwargs)


