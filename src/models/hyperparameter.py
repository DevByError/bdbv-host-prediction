"""Hyperparameter search: grid, random, and Optuna."""

import numpy as np
from typing import Dict, List, Optional, Union, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Tunes a raw estimator class by grid, random, or Optuna search.

    model_class has to be a real sklearn-style class (RandomForestClassifier,
    XGBClassifier), not one of our BaseModel wrappers - sklearn clones the
    estimator and the wrappers don't survive that. Wrap afterwards with
    to_base_model().
    """

    def __init__(
        self,
        model_class,
        param_grid: Dict,
        scoring: str = "f1",
        n_folds: int = 5,
        random_state: int = 42,
    ):
        # param_grid is name -> list of values for grid/random; Optuna also
        # accepts a (low, high, step) tuple for a numeric range
        self.model_class = model_class
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_folds = n_folds
        self.random_state = random_state
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.results = []

    def grid_search(self, X: np.ndarray, y: np.ndarray, n_jobs: int = -1) -> Dict:
        from sklearn.model_selection import GridSearchCV

        logger.info("Performing grid search...")

        model = self.model_class()

        cv = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        grid_search = GridSearchCV(
            model,
            self.param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=n_jobs,
            verbose=1,
        )

        grid_search.fit(X, y)

        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_model = grid_search.best_estimator_

        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_results": grid_search.cv_results_,
        }

    def random_search(
        self, X: np.ndarray, y: np.ndarray, n_iter: int = 100, n_jobs: int = -1
    ) -> Dict:
        from sklearn.model_selection import RandomizedSearchCV

        logger.info(f"Performing random search ({n_iter} iterations)...")

        model = self.model_class()

        cv = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        random_search = RandomizedSearchCV(
            model,
            self.param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=self.scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=self.random_state,
        )

        random_search.fit(X, y)

        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_model = random_search.best_estimator_

        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "cv_results": random_search.cv_results_,
        }

    def optuna_optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict:
        """TPE search; returns the history dict, with the study object under 'study'."""
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is not installed. Install with: pip install optuna, "
                "or use grid_search()/random_search() instead."
            )

        logger.info(f"Performing Optuna optimization ({n_trials} trials)...")

        def objective(trial):
            params = {}

            for key, value in self.param_grid.items():
                if isinstance(value, list):
                    params[key] = trial.suggest_categorical(key, value)
                elif isinstance(value, tuple) and len(value) == 3:
                    if isinstance(value[0], int):
                        params[key] = trial.suggest_int(key, value[0], value[1])
                    else:
                        params[key] = trial.suggest_float(key, value[0], value[1])
                else:
                    params[key] = value

            model = self.model_class(**params)

            cv = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )

            scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring)

            return scores.mean()

        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        self.best_params = study.best_params
        self.best_score = study.best_value

        # refit on the full X/y, the CV scores above came from folds
        self.best_model = self.model_class(**self.best_params)
        self.best_model.fit(X, y)

        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")

        history = {
            "trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "study": study,
        }

        return history

    def get_best_model(self):
        """The raw estimator, not a BaseModel wrapper."""
        return self.best_model

    def to_base_model(self, base_model_class):
        """Build (unfitted) BaseModel wrapper from best_params - caller still fits it."""
        if self.best_params is None:
            raise ValueError("Run grid_search/random_search/optuna_optimize first")
        return base_model_class(config=dict(self.best_params))
