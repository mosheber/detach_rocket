from sklearn.base import ClassifierMixin
from sklearn.linear_model import (RidgeClassifierCV ,RidgeClassifier)
import numpy as np
from sklearn.model_selection import GridSearchCV
import torch
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from skorch.callbacks import EpochTimer, PassthroughScoring, EpochScoring, PrintLog
from detach_rocket.torch_frame_trompt_encoder import FeatureImpNeuralNetClassifier, TromptModel

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
    def __init__(self, X, y, **kwargs):
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
    def __init__(self, X, y, **kwargs):
        super().__init__(X, y)
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
    def __init__(self, X, y, model_type, **kwargs):
        super().__init__(X, y)
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
    def __init__(self, X, y, **kwargs):
        super().__init__(X, y, model_type=XGBClassifier)

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

def get_first_comb(base_params):
    return {k:v[0] for k,v in base_params.items()} 


class TypeCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(torch.long)
        return super().forward(input=input, target=target)

class FeatImpWrapper(SKLearnClassifierWrapper):
    def __init__(self, X, y, **kwargs):
        ClassifierWrapper.__init__(self, X, y)
        self.X = X
        self.pos_weight = (y == 0).sum() / (y == 1).sum()
        print(f"{self.pos_weight=}")
        self.model_type = FeatureImpNeuralNetClassifier
        patch_sklearn_model(self.model_type)

        self.kwargs = kwargs

        self._full_classifier = None
        # Create the GridSearchCV object
        # base_params = self.get_grid_params()
        # base_params = {k:v[0] for k,v in base_params.items()}
        # self._full_classifier = GridSearchCV(
        #     self.model_type(**base_params), 
        #     self.get_grid_params(), 
        #     scoring='f1', 
        #     verbose=0
        # )

    def fit(self, X, y):
        pass
        # self._full_classifier.fit(X.astype(np.float32), y)

    def get_chosen_hyperparams(self) -> dict:
        self.best_params_ = get_first_comb(self.get_grid_params())
        return self.best_params_
        # self.best_params_ = self._full_classifier.best_params_

        # print('Optimal Full ROCKET: {}'.format(self.best_params_))
        # return self.best_params_
    
    def set_chosen_hyperparams(self, hyperparams: dict):
        pass
        # self.best_params_.update(hyperparams)


    def get_grid_params(self):
        # callbacks = [
        #     ('valid_f1', EpochScoring(
        #         'f1',
        #         name='valid_f1',
        #         lower_is_better=False,
        #     )),
        # ]

        params = dict(
            module=[TromptModel],
            criterion=[TypeCrossEntropyLoss(torch.tensor([1.0,self.pos_weight]))],
            max_epochs=[10],
            lr=[0.1],
            # Shuffle training data on each epoch
            iterator_train__shuffle=[True],
            # callbacks=[callbacks],

            # initialized_=False,
            module__channels=[128], # args.channels,
            module__out_channels=[2],
            module__num_prompts=[128],
            module__num_layers=[6],#args.num_layers,
            module__X_sample=[self.X],
            module__feature_importance_type=["gumbel"]
        )
        for k,v in self.kwargs.items():
            print(f"Adding {k} with values {[v]}")
            params[k] = [v]

        return params
    
    def fit_new_model_on_chosen_hyperparams(self, X_train, y_train) -> ClassifierMixin:
        params = {k:v for k,v in self.best_params_.items() if k!="module__X_sample"}
        params["module__X_sample"] = X_train
        sfd_classifier = self.model_type(**params)
        sfd_classifier.fit(X_train, y_train)
        return sfd_classifier
        
CLASSIFIER_WRAPPER_MAP = {
    "RidgeClassifier": RidgeClassifierWrapper,
    "XGBClassifier": XGBClassifierWrapper,
    "FeatImpWrapper": FeatImpWrapper
}

def create_new_classifier(name, X, y, **kwargs):
    return CLASSIFIER_WRAPPER_MAP[name](X, y, **kwargs)


