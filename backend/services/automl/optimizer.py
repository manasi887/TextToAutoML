"""Optional Bayesian hyperparameter optimization helpers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from sklearn.base import clone


MAX_OPTUNA_TRIALS = 30
MAX_RANDOM_TRIALS = 30


@dataclass(frozen=True)
class AutoOptimizerPolicyConfig:
	"""Conservative workload limits for automatic optimizer selection."""

	minimum_training_rows: int = 250
	minimum_cv_folds: int = 3
	minimum_group_count: int = 5
	minimum_bayesian_rows: int = 2000
	maximum_trials_per_model: int = 8
	maximum_total_work_units: int = 300


class OptunaDependencyError(ValueError):
	"""Raised when Bayesian tuning is selected without its optional dependency."""


def _load_optuna() -> Any:
	"""Load Optuna only when a caller explicitly requests Bayesian tuning."""
	try:
		import optuna
	except ModuleNotFoundError as exc:
		if exc.name == "optuna":
			raise OptunaDependencyError(
				"Bayesian optimization requires the optional Optuna dependency. "
				'Install it with `python -m pip install "optuna>=4.0,<5"` '
				"and retry, or select grid search."
			) from exc
		raise
	return optuna


def ensure_optuna_available() -> None:
	"""Validate the optional dependency before any Bayesian tuning prechecks."""
	_load_optuna()


def is_optuna_available() -> bool:
	"""Check whether Optuna can be imported without loading it into memory."""
	try:
		from importlib.util import find_spec
		return find_spec("optuna") is not None
	except (ImportError, ValueError):
		return False


def select_auto_optimizer(
	*,
	training_rows: int,
	feature_count: int,
	problem_type: str,
	candidate_models: list[str],
	cv_folds: int | None,
	group_count: int | None,
	optuna_available: bool,
	bayesian_models: set[str],
	random_models: set[str],
	search_space_dimensions: dict[str, int] | None = None,
	maximum_trials: int = MAX_OPTUNA_TRIALS,
	config: AutoOptimizerPolicyConfig = AutoOptimizerPolicyConfig(),
) -> dict[str, Any]:
	"""Choose a bounded tuning strategy using only known training-side facts."""
	problem = str(problem_type or "").strip().lower()
	if "classification" in problem:
		problem = "classification"
	models = list(dict.fromkeys(candidate_models))
	group_aware = group_count is not None
	base_workload = {
		"training_rows": int(training_rows),
		"feature_count": int(feature_count),
		"candidate_model_count": len(models),
		"cv_folds": int(cv_folds) if cv_folds is not None else 0,
		"group_count": int(group_count) if group_count is not None else None,
		"group_aware": group_aware,
	}

	def no_tuning(reason: str) -> dict[str, Any]:
		return {
			"requested_mode": "auto",
			"selected_strategy": "none",
			"trial_budget": 0,
			"budget_unit": "trials_per_model",
			"reason": reason,
			"workload": {
				**base_workload,
				"estimated_work_units_per_trial": 0,
				"estimated_total_work_units": 0,
			},
		}

	if problem not in {"classification", "regression"}:
		return no_tuning("Automatic tuning is limited to supervised classification and regression.")
	if not models:
		return no_tuning("No candidate models were successfully trained.")
	if training_rows < config.minimum_training_rows:
		return no_tuning(
			f"Training partition has fewer than {config.minimum_training_rows} rows; keep baseline models."
		)
	if cv_folds is None or cv_folds < config.minimum_cv_folds:
		return no_tuning(
			f"Cross-validation has fewer than {config.minimum_cv_folds} valid folds."
		)
	if group_aware and group_count < config.minimum_group_count:
		return no_tuning(
			f"Group-aware tuning needs at least {config.minimum_group_count} training groups."
		)
	unsupported_random = sorted(set(models) - random_models)
	if unsupported_random:
		return no_tuning(
			"No compatible random-search space is defined for candidate model(s): "
			+ ", ".join(unsupported_random)
			+ "."
		)

	model_cost = sum(2 if "RandomForest" in model_name else 1 for model_name in models)
	search_parameter_count = sum(
		max(0, int((search_space_dimensions or {}).get(model_name, 0)))
		for model_name in models
	)
	search_space_units = max(
		1,
		math.ceil(search_parameter_count / max(1, len(models) * 3)),
	)
	row_units = max(1, math.ceil(training_rows / 1000))
	feature_units = max(1, math.ceil(feature_count / 20))
	work_units_per_trial = (
		row_units * feature_units * cv_folds * model_cost * search_space_units
	)
	work_limited_trials = config.maximum_total_work_units // work_units_per_trial
	trial_budget = min(
		config.maximum_trials_per_model,
		maximum_trials,
		MAX_RANDOM_TRIALS,
		work_limited_trials,
	)
	if trial_budget < 2:
		return no_tuning(
			"The estimated CV workload exceeds the automatic tuning budget."
		)

	bayesian_supported = set(models).issubset(bayesian_models)
	use_bayesian = (
		optuna_available
		and bayesian_supported
		and training_rows >= config.minimum_bayesian_rows
	)
	strategy = "bayesian" if use_bayesian else "random"
	if use_bayesian:
		reason = (
			"Sufficient training data and CV capacity are available; all candidates have Bayesian "
			"search spaces and Optuna is installed."
		)
	elif not optuna_available:
		reason = "Optuna is unavailable; use reproducible Random Search with compatible model spaces."
	elif not bayesian_supported:
		reason = "At least one candidate lacks a Bayesian space; use compatible Random Search."
	else:
		reason = (
			f"Training partition is below {config.minimum_bayesian_rows} rows; use Random Search "
			"within the bounded CV workload."
		)

	return {
		"requested_mode": "auto",
		"selected_strategy": strategy,
		"trial_budget": int(trial_budget),
		"budget_unit": "trials_per_model",
		"reason": reason,
		"workload": {
			**base_workload,
			"estimated_work_units_per_trial": int(work_units_per_trial),
			"estimated_total_work_units": int(work_units_per_trial * trial_budget),
			"search_space_parameter_count": int(search_parameter_count),
			"search_space_work_units": int(search_space_units),
			"maximum_total_work_units": config.maximum_total_work_units,
		},
	}


def _suggest_parameters(trial: Any, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
	parameters: dict[str, Any] = {}
	for name, specification in search_space.items():
		kind = specification["kind"]
		if kind == "float":
			parameters[name] = trial.suggest_float(
				name,
				specification["low"],
				specification["high"],
				log=specification.get("log", False),
			)
		elif kind == "int":
			parameters[name] = trial.suggest_int(
				name,
				specification["low"],
				specification["high"],
				step=specification.get("step", 1),
				log=specification.get("log", False),
			)
		elif kind == "categorical":
			parameters[name] = trial.suggest_categorical(name, specification["choices"])
		else:
			raise ValueError(f"Unsupported Optuna search-space type '{kind}' for '{name}'.")
	return parameters


def optimize_with_optuna(
	model: Any,
	search_space: dict[str, dict[str, Any]],
	objective: Callable[[dict[str, Any]], float],
	*,
	direction: str,
	n_trials: int,
	random_state: int = 42,
) -> dict[str, Any]:
	"""Optimize estimator parameters while keeping CV evaluation caller-owned."""
	if direction not in {"minimize", "maximize"}:
		raise ValueError("direction must be either 'minimize' or 'maximize'.")
	if isinstance(n_trials, bool) or not isinstance(n_trials, int):
		raise ValueError("n_trials must be an integer.")
	if not 1 <= n_trials <= MAX_OPTUNA_TRIALS:
		raise ValueError(f"n_trials must be between 1 and {MAX_OPTUNA_TRIALS}.")
	if not search_space:
		raise ValueError("A non-empty search space is required for Bayesian optimization.")

	optuna = _load_optuna()
	study = optuna.create_study(
		direction=direction,
		sampler=optuna.samplers.TPESampler(seed=random_state),
	)

	def run_trial(trial: Any) -> float:
		parameters = _suggest_parameters(trial, search_space)
		value = float(objective(parameters))
		if not math.isfinite(value):
			raise ValueError("The tuning objective returned a non-finite metric value.")
		return value

	study.optimize(run_trial, n_trials=n_trials, show_progress_bar=False)
	best_parameters = dict(study.best_params)
	best_model = clone(model)
	estimator_parameters = best_model.get_params(deep=False)
	if "random_state" in estimator_parameters:
		best_parameters["random_state"] = random_state
	best_model.set_params(**best_parameters)
	return {
		"model": best_model,
		"parameters": best_parameters,
		"value": float(study.best_value),
		"trials_run": len(study.trials),
	}


def optimize_with_random_search(
	model: Any,
	search_space: dict[str, dict[str, Any]],
	objective: Callable[[dict[str, Any]], float],
	*,
	direction: str,
	n_trials: int,
	random_state: int = 42,
) -> dict[str, Any]:
	"""Run a reproducible bounded random search over a typed search space."""
	if direction not in {"minimize", "maximize"}:
		raise ValueError("direction must be either 'minimize' or 'maximize'.")
	if isinstance(n_trials, bool) or not isinstance(n_trials, int):
		raise ValueError("n_trials must be an integer.")
	if not 1 <= n_trials <= MAX_RANDOM_TRIALS:
		raise ValueError(f"n_trials must be between 1 and {MAX_RANDOM_TRIALS}.")
	if not search_space:
		raise ValueError("A non-empty search space is required for random search.")

	rng = np.random.default_rng(random_state)
	choices_by_name: dict[str, list[Any] | None] = {}
	finite_space_size = 1
	for name, specification in search_space.items():
		kind = specification["kind"]
		if kind == "categorical":
			choices = list(specification["choices"])
		elif kind == "int":
			choices = list(range(
				specification["low"],
				specification["high"] + 1,
				specification.get("step", 1),
			))
		elif kind == "float":
			choices = None
		else:
			raise ValueError(f"Unsupported random-search space type '{kind}' for '{name}'.")
		choices_by_name[name] = choices
		if choices is None:
			finite_space_size = 0
		elif finite_space_size:
			finite_space_size *= len(choices)

	if finite_space_size and finite_space_size <= 100_000:
		names = list(search_space)
		combinations = list(product(*(choices_by_name[name] for name in names)))
		order = rng.permutation(len(combinations))[:min(n_trials, len(combinations))]
		parameter_trials = [
			dict(zip(names, combinations[index]))
			for index in order
		]
	else:
		parameter_trials = []
		seen: set[tuple[tuple[str, Any], ...]] = set()
		attempts = 0
		while len(parameter_trials) < n_trials and attempts < n_trials * 20:
			attempts += 1
			parameters: dict[str, Any] = {}
			for name, specification in search_space.items():
				kind = specification["kind"]
				if kind == "float":
					low = float(specification["low"])
					high = float(specification["high"])
					if specification.get("log", False):
						value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
					else:
						value = float(rng.uniform(low, high))
				elif kind == "int":
					values = choices_by_name[name]
					value = values[int(rng.integers(len(values)))]
				else:
					values = choices_by_name[name]
					value = values[int(rng.integers(len(values)))]
				parameters[name] = value
			key = tuple(sorted(parameters.items()))
			if key not in seen:
				seen.add(key)
				parameter_trials.append(parameters)

	best_parameters: dict[str, Any] | None = None
	best_value: float | None = None
	for parameters in parameter_trials:
		value = float(objective(parameters))
		if not math.isfinite(value):
			raise ValueError("The tuning objective returned a non-finite metric value.")
		is_better = (
			best_value is None
			or (direction == "maximize" and value > best_value)
			or (direction == "minimize" and value < best_value)
		)
		if is_better:
			best_value = value
			best_parameters = parameters
	if best_parameters is None or best_value is None:
		raise ValueError("Random Search did not produce any parameter trials.")

	best_model = clone(model)
	model_parameters = dict(best_parameters)
	if "random_state" in best_model.get_params(deep=False):
		best_model.set_params(random_state=random_state)
	best_model.set_params(**model_parameters)
	return {
		"model": best_model,
		"parameters": model_parameters,
		"value": float(best_value),
		"trials_run": len(parameter_trials),
	}
