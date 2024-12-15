from sklearn.base import ClassifierMixin
from sklearn.linear_model import (RidgeClassifierCV ,RidgeClassifier)
import numpy as np
from sklearn.model_selection import GridSearchCV
import torch
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

def patch_sklearn_model(sklearn_type):
    @property
    def get_coef_(self):
        return self.feature_importances_[None,:]

    def get_score(self, X, y, sample_weight=None):
        return f1_score(y_true=y, y_pred=self.predict(X), sample_weight=sample_weight)

    sklearn_type.coef_ = get_coef_
    sklearn_type.score = get_score

def ridge_predict_proba(self, X):
    d = self.decision_function(X)
    if len(d.shape) == 1:
        d = np.c_[-d, d]
    d_exp = np.exp(d)
    return d_exp / d_exp.sum(axis=-1, keepdims=True)

RidgeClassifier.predict_proba = ridge_predict_proba

class ClassifierWrapper:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        self._full_classifier.fit(X, y)

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

    def get_chosen_hyperparams(self) -> dict:
        # if not hasattr(self, "_full_model_alpha"):
        self._full_model_alpha = self._full_classifier.alpha_

        print('Optimal Alpha Full ROCKET: {:.2f}'.format(self._full_model_alpha))
        return {
            "alpha": self._full_model_alpha
        }
    
    def set_chosen_hyperparams(self, hyperparams: dict):
        if "alpha" in hyperparams:
            self._full_model_alpha = hyperparams["alpha"]

    def fit_new_model_on_chosen_hyperparams(self, X_train, y_train) -> ClassifierMixin:
        sfd_classifier = RidgeClassifier(alpha=self._full_model_alpha)
        sfd_classifier.fit(X_train, y_train)
        return sfd_classifier


class SKLearnClassifierWrapper(ClassifierWrapper):
    def __init__(self, model_type, **kwargs):
        super().__init__()
        self.model_type = model_type
        patch_sklearn_model(model_type)

        # Create the GridSearchCV object
        self._full_classifier = GridSearchCV(
            model_type(), 
            self.get_grid_params(), 
            scoring='f1', 
            verbose=10
        )

    def get_grid_params(self) -> dict:
        return {}
        
    def get_chosen_hyperparams(self) -> dict:
        self.best_params_ = self._full_classifier.best_params_

        print('Optimal Full ROCKET: {}'.format(self.best_params_))
        return self.best_params_
    
    def set_chosen_hyperparams(self, hyperparams: dict):
        self.best_params_.update(hyperparams)

    def fit_new_model_on_chosen_hyperparams(self, X_train, y_train) -> ClassifierMixin:
        sfd_classifier = self.model_type(**self.best_params_)
        sfd_classifier.fit(X_train, y_train)
        return sfd_classifier

class XGBClassifierWrapper(SKLearnClassifierWrapper):
    def __init__(self, **kwargs):
        super().__init__(model_type=XGBClassifier)

    def get_grid_params(self):
        # Define the hyperparameter grid
        params = {}
        if torch.cuda.is_available():
            params.update({
                'device': ["cuda"],
                'tree_method': ["hist"]
            })

        params.update({
            'max_depth': [7, 10],
            # 'learning_rate': [0.1, 0.001],
            # 'subsample': [0.8, 1],
            'n_estimators': [500, 1000]
        })
        return params
        
CLASSIFIER_WRAPPER_MAP = {
    "RidgeClassifier": RidgeClassifierWrapper,
    "XGBClassifier": XGBClassifierWrapper
}

def create_new_classifier(name, **kwargs):
    return CLASSIFIER_WRAPPER_MAP[name](**kwargs)


