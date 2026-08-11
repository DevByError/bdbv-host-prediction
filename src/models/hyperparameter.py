"""
Hyperparameter optimization with Optuna and GridSearch.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from optuna.samplers import TPESampler
import logging

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization with multiple methods.

    Methods:
    - Grid Search (exhaustive)
    - Random Search
    - Bayesian Optimization (Optuna)
    """

    def __init__(
        self,
        model_class,
        param_grid: Dict,
        scoring: str = "f1",
        n_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize optimizer.

        Args:
            model_class: Model class (not instantiated)
            param_grid: Parameter grid or search space
            scoring: Scoring metric
            n_folds: Number of CV folds
            random_state: Random seed
        """
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
        """
        Perform grid search.

        Args:
            X: Features
            y: Labels
            n_jobs: Number of parallel jobs

        Returns:
            Dict: Best parameters and score
        """
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
        """
        Perform random search.

        Args:
            X: Features
            y: Labels
            n_iter: Number of iterations
            n_jobs: Number of parallel jobs

        Returns:
            Dict: Best parameters and score
        """
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
        """
        Perform Bayesian optimization with Optuna.

        Args:
            X: Features
            y: Labels
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Dict: Best parameters and score
        """
        logger.info(f"Performing Optuna optimization ({n_trials} trials)...")

        def objective(trial):
            # Sample parameters from the search space
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

            # Train model
            model = self.model_class(**params)

            # Cross-validate
            cv = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.random_state
            )

            scores = cross_val_score(model, X, y, cv=cv, scoring=self.scoring)

            return scores.mean()

        # Create study
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Get best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value

        # Train best model
        self.best_model = self.model_class(**self.best_params)
        self.best_model.fit(X, y)

        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")

        # Get optimization history
        history = {
            "trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "study": study,
        }

        return history

    def get_best_model(self):
        """Get the best model found."""
        return self.best_model
