from .old_bayesopt import BayesianOptimizer
from .logging_config import configure_logging
from .mpc import MPCController

# Auto-configure structured logging when package is imported
configure_logging()