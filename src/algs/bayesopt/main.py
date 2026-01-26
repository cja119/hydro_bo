"""
Bayesian Optimization wrapper for H2-Gym environments using Ray Tune
"""

import json
import yaml
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Dict, List, Optional, Union

import pandas as pd
import ray
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

from envs.shipping.utils import calculate_capex_opex
from .utils import ensure_dirs, parse_memory_string



class BayesianOptimizer:
    """
    Ray-based Bayesian optimization wrapper for the shipping environment.
    """

    def __init__(
        self,
        tmp_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        resume_from_checkpoint: bool = False,
    ):
        """
        Initialize the Bayesian optimizer.

        :param tmp_dir: Temporary directory for results.
        :type tmp_dir: Optional[Union[str, Path]]
        :param checkpoint_dir: Directory for saving or loading checkpoints.
        :type checkpoint_dir: Optional[Union[str, Path]]
        :param resume_from_checkpoint: Whether to resume from the latest checkpoint.
        :type resume_from_checkpoint: bool
        """
        self.tmp_dir, self.bayesopt_dir, self.ray_results_dir, self.checkpoint_dir = (
            ensure_dirs(tmp_dir, checkpoint_dir)
        )

        # Configuration parameters (set via context manager)
        self._config = {
            "env_version": "shipping",
            "vector_type": "LH2",
            "renewables_type": "wind",
            "weather_file": "CoastalChile_15-20_Wind.csv",
            "conversion_trains": 10,
            "planning_results_file": None,
            "scale": (0.5, 2.0),  # (lower_multiplier, upper_multiplier)
            "use_planning_bounds": False,
            "search_space": None,  # Will be set in __exit__
            "solver": "gurobi",
        }

        # Default search space (will be created lazily to avoid import-time issues)
        self._default_search_space = None

        # Ray setup
        self._ray_initialized = False
        self.results = None
        self._configured = False

        # Checkpoint management
        self.resume_from_checkpoint = resume_from_checkpoint
        self._checkpoint_state = None
        self._trial_history = []

        # Resource management
        self._resource_config = {
            "num_cpus": 8,
            "num_gpus": 0,
            "resources_per_trial": {"cpu": 1, "memory": None, "gpu": 0},
            "max_concurrent_trials": 8,
        }

        # Loaded results management
        self._loaded_results = None
        self._loaded_experiment_name = None

    # --- Context manager methods ---

    def __enter__(self):
        """
        Enter the context manager and return the config dictionary for modification.

        :return: Mutable configuration dictionary.
        :rtype: Dict
        """
        return self._config

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and finalize configuration.
        """
        # Generate search space based on configuration
        if (
            self._config["use_planning_bounds"]
            and self._config["planning_results_file"]
        ):
            try:
                self._config["search_space"] = self._generate_bounds_from_file(
                    self._config["planning_results_file"], self._config["scale"]
                )
                print(
                    f"Generated search space from planning file with scale {self._config['scale']}"
                )
            except Exception as e:
                print(f"Warning: Failed to generate bounds from planning file ({e})")
                print("Using default search space")
                self._config["search_space"] = self._get_default_search_space()
        else:
            self._config["search_space"] = self._get_default_search_space()

        self._configured = True

    # --- Public methods ---

    def optimize(
        self,
        num_samples: int = 50,
        n_cores: Optional[int] = None,
        experiment_name: str = "shipping_bayesopt",
        ray_config: Optional[Dict] = None,
        enable_checkpointing: bool = True,
    ) -> ray.tune.ExperimentAnalysis:
        """
        Run Bayesian optimization.

        :param num_samples: Number of optimization trials.
        :type num_samples: int
        :param n_cores: Number of CPU cores to use.
        :type n_cores: Optional[int]
        :param experiment_name: Name for the experiment.
        :type experiment_name: str
        :param ray_config: Additional Ray initialization options.
        :type ray_config: Optional[Dict]
        :param enable_checkpointing: Whether to enable checkpointing during optimization.
        :type enable_checkpointing: bool
        :return: Ray Tune analysis with results.
        :rtype: ray.tune.ExperimentAnalysis
        """
        # Store num_samples in config for resource usage tracking
        self._config["num_samples"] = num_samples

        # Initialize Ray if not already done
        if not self._ray_initialized:
            ray_init_config = {
                "num_cpus": self._resource_config["num_cpus"],
                "num_gpus": self._resource_config["num_gpus"],
            }
            if ray_config:
                ray_init_config.update(ray_config)
            ray.init(**ray_init_config)
            self._ray_initialized = True

        # Setup algorithm
        algo = OptunaSearch(metric="score", mode="max")

        # Set concurrency limit
        if n_cores is not None:
            self._setup_distributed(n_cores=n_cores)
        else:
            # Auto-detect and use all cores minus 2
            self._setup_distributed()
        max_concurrent = self._resource_config["max_concurrent_trials"]
        algo = ConcurrencyLimiter(algo, max_concurrent=max_concurrent)

        # Setup scheduler
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="score",
            mode="max",
            max_t=100,  # Maximum iterations per trial
            grace_period=1,  # Minimum iterations before early stopping
            reduction_factor=3,  # Factor by which to reduce tber of trials
        )

        if not self._configured:
            raise ValueError("Optimizer not configured. Use context manager first.")

        # Run optimization
        self.results = tune.run(
            self._objective_function,
            name=experiment_name,
            search_alg=algo,
            scheduler=scheduler,
            config=self._config["search_space"],
            num_samples=num_samples,
            resources_per_trial={"cpu": 1},
            storage_path=str(self.ray_results_dir),
        )

        return self.results

    def get_best_result(self) -> Dict:
        """
        Get the best optimization result.

        :return: Best parameters and score.
        :rtype: Dict
        """
        if self.results is None:
            raise ValueError("No optimization results available. Run optimize() first.")

        best_trial = self.results.get_best_trial("score", "max")
        return {
            "config": best_trial.config,
            "score": best_trial.last_result["score"],
            "trial_id": best_trial.last_result.get("trial_id"),
        }

    def get_results_dataframe(self):
        """
        Get optimization results as a pandas DataFrame.

        :return: Trial results DataFrame.
        :rtype: pd.DataFrame
        """
        if self.results is None and self._loaded_results is None:
            raise ValueError(
                "No optimization results available. Run optimize() or load_results() first."
            )

        if self._loaded_results is not None:
            return self._loaded_results

        return self.results.dataframe()

    def load_results(
        self, experiment_path: str, experiment_name: Optional[str] = None
    ) -> None:
        """
        Load results from a Ray Tune experiment directory.

        :param experiment_path: Path to a Ray results directory.
        :type experiment_path: str
        :param experiment_name: Specific experiment name to load.
        :type experiment_name: Optional[str]
        """
        experiment_path = Path(experiment_path)

        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment path not found: {experiment_path}")

        if experiment_name:
            # Load specific experiment
            exp_dir = experiment_path / experiment_name
            if not exp_dir.exists():
                raise FileNotFoundError(f"Experiment not found: {experiment_name}")
        else:
            # Find most recent experiment
            exp_dirs = [
                d
                for d in experiment_path.iterdir()
                if d.is_dir() and d.name.startswith("shipping")
            ]
            if not exp_dirs:
                raise FileNotFoundError("No experiments found in directory")
            exp_dir = max(exp_dirs, key=lambda x: x.stat().st_mtime)
            experiment_name = exp_dir.name

        # Load experiment analysis
        self.results = ray.tune.ExperimentAnalysis(str(exp_dir))
        self._loaded_experiment_name = experiment_name

        # Cache dataframe
        self._loaded_results = self.results.dataframe()

        print(f"Loaded experiment: {experiment_name}")
        print(f"Total trials: {len(self._loaded_results)}")

    def get_top_n_results(self, n: int = 5, metric: str = "score") -> List[Dict]:
        """
        Get the top N best results from the optimization.

        :param n: Number of top results to return.
        :type n: int
        :param metric: Metric to sort by.
        :type metric: str
        :return: Ranked result summaries.
        :rtype: List[Dict]
        """
        df = self.get_results_dataframe()

        # Sort by metric (assuming minimization)
        sorted_df = df.sort_values(by=metric).head(n)

        results = []
        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            # Extract config parameters
            config = {}
            for col in df.columns:
                if col.startswith("config/"):
                    param_name = col.replace("config/", "")
                    config[param_name] = row[col]

            # Find associated planning file
            trial_id = row.get("trial_id", idx)
            planning_file = self.bayesopt_dir / f"{trial_id}.yml"

            results.append(
                {
                    "rank": i + 1,
                    "score": row[metric],
                    "trial_id": trial_id,
                    "config": config,
                    "planning_file": (
                        str(planning_file) if planning_file.exists() else None
                    ),
                }
            )

        return results

    def export_results(
        self, format: str = "json", output_path: str = "./results"
    ) -> str:
        """
        Export optimization results in various formats.

        :param format: Export format (json, csv, yaml, excel).
        :type format: str
        :param output_path: Output file path.
        :type output_path: str
        :return: Path to the exported file.
        :rtype: str
        """
        output_path = Path(output_path)
        df = self.get_results_dataframe()

        # Prepare data for export
        export_data = {
            "experiment_name": self._loaded_experiment_name or "current",
            "total_trials": len(df),
            "best_score": df["score"].max(),
            "timestamp": datetime.now().isoformat(),
        }

        if format == "json":
            if not str(output_path).endswith(".json"):
                output_path = output_path.with_suffix(".json")

            export_data["trials"] = df.to_dict(orient="records")
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format == "csv":
            if not str(output_path).endswith(".csv"):
                output_path = output_path.with_suffix(".csv")
            df.to_csv(output_path, index=False)

        elif format == "yaml":
            if not str(output_path).endswith(".yml") and not str(output_path).endswith(
                ".yaml"
            ):
                output_path = output_path.with_suffix(".yml")

            export_data["trials"] = df.to_dict(orient="records")
            with open(output_path, "w") as f:
                yaml.dump(export_data, f)

        elif format == "excel":
            if not str(output_path).endswith(".xlsx"):
                output_path = output_path.with_suffix(".xlsx")

            with pd.ExcelWriter(output_path) as writer:
                # Summary sheet
                summary_df = pd.DataFrame([export_data])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # Results sheet
                df.to_excel(writer, sheet_name="Results", index=False)

                # Top 10 results
                top_10 = self.get_top_n_results(n=10)
                top_df = pd.DataFrame(top_10)
                top_df.to_excel(writer, sheet_name="Top 10", index=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(output_path)

    def cleanup(self):
        """
        Clean up Ray resources.
        """
        if self._ray_initialized:
            ray.shutdown()
            self._ray_initialized = False

    # --- Helper methods ---

    def _setup_distributed(
        self,
        n_cores: Optional[int] = None,
        memory_per_trial: str = "2GB",
        gpu_per_trial: float = 0.0,
    ) -> Dict:
        """
        Configure distributed computing resources.

        :param n_cores: Number of CPU cores to use; defaults to all minus two.
        :type n_cores: Optional[int]
        :param memory_per_trial: Memory allocation per trial (e.g., "2GB", "512MB").
        :type memory_per_trial: str
        :param gpu_per_trial: GPU allocation per trial (fractional GPUs supported).
        :type gpu_per_trial: float
        :return: Updated resource configuration.
        :rtype: Dict
        """
        if n_cores is None:
            total_cores = self._resource_config["num_cpus"]
            n_cores = max(1, total_cores - 2)

        memory_bytes = parse_memory_string(memory_per_trial)

        self._resource_config = {
            "num_cpus": self._resource_config["num_cpus"],
            "num_gpus": int(gpu_per_trial * n_cores) if gpu_per_trial > 0 else 0,
            "resources_per_trial": {
                "cpu": 1,
                "memory": memory_bytes,
                "gpu": gpu_per_trial,
            },
            "max_concurrent_trials": self._resource_config["max_concurrent_trials"],
        }

        return self._resource_config.copy()
    
    def _objective_function(self, config: Dict) -> None:
        """
        Objective function for Ray Tune optimization.

        :param config: Hyperparameters sampled by Ray Tune.
        :type config: Dict
        """
        if not self._configured:
            raise ValueError("Optimizer not configured. Use context manager first.")

        # Extract parameters
        args = [
            config["compression_capacity"],
            config["electrolyser_capacity"],
            config["fuelcell_capacity"],
            config["hydrogen_storage_capacity"],
            config["renewable_energy_capacity"],
            config["vector_storage_capacity"],
        ]

        # Generate unique ID for this trial
        trial_id = randint(0, 1000000)

        # Calculate CAPEX/OPEX parameters
        results = calculate_capex_opex(
            renewables=self._config["renewables_type"],
            vector=self._config["vector_type"],
            compression_capacity=args[0],
            electrolyser_capacity=args[1],
            fuelcell_capacity=args[2],
            conversion_trains_number=self._config["conversion_trains"],
            hydrogen_storage_capacity=args[3],
            renewable_energy_capacity=args[4],
            vector_storage_capacity=args[5],
        )

        # Save planning parameters
        planning_file = self.bayesopt_dir / f"{trial_id}.yml"
        with open(planning_file, "w") as yml_file:
            yaml.dump(results, yml_file)

        # Run shipping environment
        from hydro_bo.envs import ShippingEnv

        environment = ShippingEnv(self._config["env_version"])

        with environment as env:
            env["vector"] = self._config["vector_type"]
            env["fast"]["data_folder"] = "version2"
            env["fast"]["solver"] = self._config["solver"]
            env["fast"]["planning_model"] = str(planning_file)
            env["weather_data"]["weather_file"] = self._config["weather_file"]

        # Execute environment and collect rewards
        total_reward = 0
        total_tonnes = 0
        while True:
            observation, reward, done, _ = environment.step(action=None, verbose=False)
            total_reward += reward
            total_tonnes += observation["total_tonnes"]
            if done:
                break

        # Report result to Ray (negative because Ray minimizes)
        session.report({"score": total_reward / total_tonnes, "trial_id": trial_id})