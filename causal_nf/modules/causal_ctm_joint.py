import torch
import torch.nn as nn

from torch import Tensor, Size

from zuko.transforms import ComposedTransform


class CausalCTMJoint(nn.Module):
    """
    Joint Causal CTM that models the complete vector Y with a single conditional flow.

    Historically this class was named `CausalCTM`. It has been renamed to
    `CausalCTMJoint` to distinguish it from the node-wise implementation located in
    `causal_ctm_nodewise.py`.
    """

    def __init__(self, flow, context_dim: int = 0):
        super().__init__()
        self.flow = flow
        self.adjacency = None
        self.context_dim = int(context_dim) if context_dim is not None else 0

    # ------------------------------------------------------------------ #
    # Graph utilities                                                    #
    # ------------------------------------------------------------------ #
    def set_adjacency(self, adj: Tensor):
        self.adjacency = adj

    # ------------------------------------------------------------------ #
    # Context helpers                                                    #
    # ------------------------------------------------------------------ #
    def _ensure_ctx(self, y: Tensor, x_ctx: Tensor = None) -> Tensor:
        if self.context_dim == 0:
            return None

        if x_ctx is None:
            return torch.zeros(
                (y.shape[0], self.context_dim), device=y.device, dtype=y.dtype
            )
        return x_ctx

    def _flow_device(self):
        try:
            return next(self.flow.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # Core API                                                           #
    # ------------------------------------------------------------------ #
    def forward(self, y: Tensor, x_ctx: Tensor = None) -> dict:
        ctx = self._ensure_ctx(y, x_ctx)
        n_flow = self.flow(ctx)
        log_prob = n_flow.log_prob(y)
        return {"log_prob": log_prob, "loss": -log_prob}

    def log_prob(self, y: Tensor, x_ctx: Tensor = None, scaler=None) -> Tensor:
        ctx = self._ensure_ctx(y, x_ctx)
        n_flow = self.flow(ctx)
        if scaler is not None:
            n_flow.transform = ComposedTransform(scaler, n_flow.transform)
        return n_flow.log_prob(y)

    def sample_conditional(self, x_ctx: Tensor, shape: Size = ()) -> dict:
        n_flow = self.flow(x_ctx)
        y, u = n_flow.sample_u(shape)
        return {"u_obs": u, "y_obs": y}

    # ------------------------------------------------------------------ #
    # Interventions & counterfactuals                                    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_counterfactual(
        self,
        y_factual: Tensor,
        index: int,
        value: float,
        x_ctx: Tensor = None,
        scaler=None,
        return_dict: bool = False,
    ):
        ctx = self._ensure_ctx(y_factual, x_ctx)
        n_flow = self.flow(ctx)
        if scaler is not None:
            n_flow.transform = ComposedTransform(scaler, n_flow.transform)

        z_factual = n_flow.transform(y_factual)
        y_modified = y_factual.clone()
        y_modified[:, index] = value
        z_cf = n_flow.transform(y_modified)
        z_factual[:, index] = z_cf[:, index]
        y_cf = n_flow.transform.inv(z_factual)

        if return_dict:
            return {"y_cf": y_cf, "z_factual": z_factual, "z_cf": z_cf}
        return y_cf

    @torch.no_grad()
    def intervene(
        self,
        index: int,
        value: float,
        x_ctx: Tensor = None,
        shape: Size = (),
        scaler=None,
    ) -> Tensor:
        if isinstance(shape, tuple):
            n = shape[0] if len(shape) > 0 else 1
        else:
            n = shape[0]

        device = x_ctx.device if x_ctx is not None else self._flow_device()
        dummy_y = torch.zeros((n, 1), device=device)
        ctx = self._ensure_ctx(dummy_y, x_ctx)
        n_flow = self.flow(ctx)
        if scaler is not None:
            n_flow.transform = ComposedTransform(scaler, n_flow.transform)

        z_base = n_flow.base.rsample(shape)
        y_sample = n_flow.transform.inv(z_base)
        y_sample[:, index] = value
        z_updated = n_flow.transform(y_sample)
        z_updated[:, index + 1 :] = z_base[:, index + 1 :]
        return n_flow.transform.inv(z_updated)

    @torch.no_grad()
    def do_x(
        self,
        y_factual: Tensor,
        x_factual: Tensor,
        x_tilde: Tensor,
        scaler=None,
    ) -> Tensor:
        ctx_f = self._ensure_ctx(y_factual, x_factual)
        n_flow_f = self.flow(ctx_f)
        if scaler is not None:
            n_flow_f.transform = ComposedTransform(scaler, n_flow_f.transform)
        z_f = n_flow_f.transform(y_factual)

        ctx_t = self._ensure_ctx(y_factual, x_tilde)
        n_flow_t = self.flow(ctx_t)
        if scaler is not None:
            n_flow_t.transform = ComposedTransform(scaler, n_flow_t.transform)
        return n_flow_t.transform.inv(z_f)
