import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from types import SimpleNamespace
from symbolicregression.envs import build_env
import symbolicregression
from tpsr import tpsr_fit
from symbolicregression.e2e_model import pred_for_sample_no_refine
from symbolicregression.model.sklearn_wrapper import get_top_k_features
import symbolicregression.model.utils_wrapper as utils_wrapper


class TPSRRegressor(BaseEstimator, RegressorMixin):
    """Compact sklearn wrapper for TPSR (Transformer-based Planning for Symbolic Regression)."""

    def __init__(
        self,
        max_input_points=200,
        width=3,
        horizon=200,
        rollout=3,
        num_beams=1,
        lam=0.1,
        ucb_constant=1.0,
        ucb_base=10.0,
        uct_alg="uct",
        rescale=True,
        device=None,
        **kwargs,
    ):
        self.max_input_points = max_input_points
        self.width = width
        self.horizon = horizon
        self.rollout = rollout
        self.num_beams = num_beams
        self.lam = lam
        self.ucb_constant = ucb_constant
        self.ucb_base = ucb_base
        self.uct_alg = uct_alg
        self.rescale = rescale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, X, y, verbose=False):
        """Fit TPSR model to data."""
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Feature selection
        self.top_k_features = get_top_k_features(X, y, k=min(10, X.shape[1]))
        X = X[:, self.top_k_features]

        # Scaling
        self.scaler = utils_wrapper.StandardScaler() if self.rescale else None
        if self.scaler:
            X = self.scaler.fit_transform(X)
            self.scale_params = self.scaler.get_params()
        else:
            self.scale_params = None

        # Setup params
        params = self._create_params()

        # Build environment
        equation_env = build_env(params)
        symbolicregression.utils.CUDA = params.device != "cpu"

        # Run TPSR
        scaled_X = [X]
        Y = [y]
        self.equation_sequence, self.fit_time, self.sample_times = tpsr_fit(
            scaled_X, Y, params, equation_env, bag_number=1, rescale=False
        )

        # Store for prediction
        self.equation_env = equation_env
        self.params = params
        self._create_model_for_prediction(X, y)

        return self

    def predict(self, X):
        """Predict using the fitted TPSR model."""
        X = X[:, self.top_k_features]
        if self.scaler:
            X = self.scaler.transform(X)

        y_pred, _, _ = pred_for_sample_no_refine(
            self.model, self.equation_env, self.equation_sequence, [X]
        )

        if y_pred is None:
            return np.zeros(X.shape[0])
        return y_pred

    def _create_params(self):
        """Create params namespace with defaults."""
        params = SimpleNamespace()
        params.backbone_model = "e2e"
        params.max_input_points = self.max_input_points
        params.width = self.width
        params.horizon = self.horizon
        params.rollout = self.rollout
        params.num_beams = self.num_beams
        params.lam = self.lam
        params.ucb_constant = self.ucb_constant
        params.ucb_base = self.ucb_base
        params.uct_alg = self.uct_alg
        params.device = torch.device(self.device)
        params.no_seq_cache = False
        params.no_prefix_cache = True
        params.beam_length_penalty = 1.0
        params.beam_size = 1
        params.beam_type = "search"
        params.beam_early_stopping = True
        params.beam_temperature = 1.0
        params.max_generated_output_len = 200
        params.n_trees_to_refine = 1
        params.max_number_bags = 1
        params.rescale = self.rescale
        params.train_value = False
        params.debug = False
        params.sample_only = False
        params.cpu = self.device == "cpu"
        params.reload_model = "./symbolicregression/weights/model.pt"
        params.env_name = "functions"
        params.tasks = "functions"
        params.max_input_dimension = 10
        params.float_precision = 10
        params.mantissa_len = 11  # Must divide (float_precision + 1) = 11
        # Generator parameters
        params.prob_const = 0.0
        params.prob_rand = 0.0
        params.max_int = 10
        params.min_binary_ops_per_dim = 0
        params.max_binary_ops_per_dim = 1
        params.min_unary_ops = 0
        params.max_unary_ops = 4
        params.min_output_dimension = 1
        params.min_input_dimension = 1
        params.max_output_dimension = 1
        params.max_exponent = 100
        params.max_exponent_prefactor = 1
        params.operators_to_downsample = (
            "div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3"
        )
        params.required_operators = ""
        params.extra_binary_operators = ""
        params.extra_unary_operators = ""
        params.extra_constants = None
        return params

    def _create_model_for_prediction(self, X, y):
        """Create model instance for prediction."""
        from symbolicregression.e2e_model import Transformer

        samples = {"x_to_fit": [X], "y_to_fit": [y], "x_to_pred": [X], "y_to_pred": [y]}
        self.model = Transformer(
            params=self.params, env=self.equation_env, samples=samples
        )
        self.model.to(self.params.device)
        self.model.eval()
