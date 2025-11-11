import torch
import torch.nn as nn

from torch import Tensor, Size
from zuko.transforms import ComposedTransform


class CausalCTM(nn.Module):
    """
    Strict 'one CTM per node' implementation.

    - We assume a DAG over d variables (y_1, ..., y_d).
    - For each node i, we have a 1D conditional flow (node_flows[i]) that models
      p(y_i | y_pa(i)), where pa(i) are parents from the adjacency matrix.

    node_flows[i] must be a *conditional* flow factory:
        node_flows[i](ctx) -> n_flow_i

    where n_flow_i has:
        - .log_prob(y_i) for y_i of shape (N, 1)
        - .base.rsample(shape) for exogenous u_i
        - .transform(y_i) and .transform.inv(u_i)
    """

    def __init__(self, node_flows, context_dim_ext: int = 0):
        """
        Args:
            node_flows: nn.ModuleList (or list) of length d.
                Each element is a conditional flow object `flow_i` such that
                `flow_i(ctx_i)` returns a zuko-style flow with:
                    log_prob(y_i), base.rsample(shape), transform, transform.inv
                where y_i has shape (N, 1) and ctx_i has shape (N, |pa(i)| + context_dim_ext) or None.

            context_dim_ext: optional external context dimension to append to parents.
                If you don't use external context, set this to 0.
        """
        super().__init__()
        # node_flows[i] is a callable/Flow factory
        self.node_flows = nn.ModuleList(node_flows)
        self.adjacency = None
        self.parents = None          # list of LongTensors with parent indices per node
        self.topo_order = None       # list[int], topological order of nodes
        self.context_dim_ext = int(context_dim_ext) if context_dim_ext is not None else 0

    # -------------------------------------------------------------------------
    # Graph structure
    # -------------------------------------------------------------------------

    def set_adjacency(self, adj: Tensor):
        """
        adj: (d, d) tensor with adj[i, j] = 1 if i -> j.
        We assume adj is acyclic.
        """
        self.adjacency = adj.clone().detach()
        d = adj.shape[0]

        # parents[i] = indices j such that j -> i
        parents = []
        for i in range(d):
            pa_i = torch.where(self.adjacency[:, i] != 0)[0]
            parents.append(pa_i)
        self.parents = parents

        # compute a topological ordering
        self.topo_order = self._topological_order(self.adjacency)

    @staticmethod
    def _topological_order(adj: Tensor):
        """
        Kahn's algorithm for topological sorting of a DAG.
        adj[i,j] = 1 if i -> j.
        """
        d = adj.shape[0]
        in_deg = adj.sum(0).clone().long()  # indegree per node
        order = []
        queue = [i for i in range(d) if in_deg[i] == 0]

        while queue:
            i = queue.pop(0)
            order.append(i)
            # remove edges i -> j
            for j in range(d):
                if adj[i, j] != 0:
                    in_deg[j] -= 1
                    if in_deg[j] == 0:
                        queue.append(j)

        if len(order) != d:
            raise RuntimeError("Adjacency matrix is not acyclic; topological sort failed.")
        return order

    # -------------------------------------------------------------------------
    # Context handling
    # -------------------------------------------------------------------------

    def _build_ctx_i(self, y: Tensor, i: int, x_ext: Tensor = None) -> Tensor:
        """
        Build context for node i:
            ctx_i = [ y_pa(i) , x_ext ]  (if any parents / external context exist)
        """
        pa_i = self.parents[i]
        pieces = []

        if pa_i.numel() > 0:
            pieces.append(y[:, pa_i])  # (N, |pa(i)|)

        if self.context_dim_ext > 0:
            if x_ext is None:
                # external context not provided -> zeros
                N = y.shape[0]
                pieces.append(
                    torch.zeros(N, self.context_dim_ext, device=y.device, dtype=y.dtype)
                )
            else:
                # assume x_ext.shape[0] == N and last dim == context_dim_ext
                pieces.append(x_ext)

        if not pieces:
            return None  # no context
        else:
            return torch.cat(pieces, dim=-1)

    # -------------------------------------------------------------------------
    # Core methods
    # -------------------------------------------------------------------------

    def forward(self, y: Tensor, x_ctx: Tensor = None) -> dict:
        """
        Compute joint log_prob(y) = sum_i log p(y_i | y_pa(i), x_ctx).
        """
        output = {}
        log_prob = self.log_prob(y, x_ctx=x_ctx, scaler=None)
        output["log_prob"] = log_prob
        output["loss"] = -log_prob
        return output

    def log_prob(self, y: Tensor, x_ctx: Tensor = None, scaler=None) -> Tensor:
        """
        Joint log_prob under the node-wise CTM:
            log p(y) = sum_i log p(y_i | y_pa(i), x_ctx).
        """
        if scaler is not None:
            # We compose a global scaler as in your original code:
            # y_scaled = scaler(y); but we also need to scale per-node consistently.
            # Easiest: push scaler into each node's transform only once per call
            # by wrapping y before passing into per-node flows.
            y = scaler(y)

        N, d = y.shape
        assert d == len(self.node_flows), "y dimension must match number of node_flows."

        total_log_prob = torch.zeros(N, device=y.device, dtype=y.dtype)
        for i in range(d):
            # Build context for node i
            ctx_i = self._build_ctx_i(y, i, x_ext=x_ctx)  # x_ctx is external context
            flow_i = self.node_flows[i](ctx_i)            # conditional flow for node i
            y_i = y[:, i:i+1]                             # shape (N, 1)
            log_prob_i = flow_i.log_prob(y_i)             # shape (N,)
            total_log_prob = total_log_prob + log_prob_i

        return total_log_prob

    def sample_conditional(self, x_ctx: Tensor = None, shape: Size = ()) -> dict:
        """
        Sample y ~ p(y | x_ctx) using node-wise CTMs in topological order.

        Returns dict with:
            - "y_obs": sampled y (N, d)
            - "u_obs": exogenous latents concatenated (N, d)
        """
        output = {}
        if isinstance(shape, tuple):
            if len(shape) == 0:
                N = 1
            else:
                N = shape[0]
        else:
            N = shape[0]

        d = len(self.node_flows)
        y = torch.zeros(N, d)
        u_all = torch.zeros(N, d)

        # We sample node-wise exogenous variables independently and transform them
        # to y_i given parents
        for i in self.topo_order:
            # context_i uses already-sampled parents in y
            ctx_i = self._build_ctx_i(y, i, x_ext=x_ctx)
            flow_i = self.node_flows[i](ctx_i)

            # sample exogenous u_i from base
            if ctx_i is None:
                u_i = flow_i.base.rsample((N,))
            else:
                u_i = flow_i.base.rsample()
            u_i = u_i.reshape(N, -1)
            # map to y_i via inverse transform
            y_i = flow_i.transform.inv(u_i)
            y_i = y_i.reshape(N, -1)

            y[:, i] = y_i[:, 0]
            u_all[:, i] = u_i[:, 0]

        output["y_obs"] = y
        output["u_obs"] = u_all
        return output

    @torch.no_grad()
    def intervene(
        self,
        index: int,
        value: float,
        x_ctx: Tensor = None,
        shape: Size = (),
        scaler=None,
    ) -> Tensor:
        """
        do(Y_index = value).

        We implement SCM semantics:
        - For node index: replace its structural equation by constant value.
        - For others: use their CTM with parents (some of which may now include the intervened node).
        """
        if isinstance(shape, tuple):
            if len(shape) == 0:
                N = 1
            else:
                N = shape[0]
        else:
            N = shape[0]

        d = len(self.node_flows)
        device = x_ctx.device if x_ctx is not None else None
        y = torch.zeros(N, d, device=device)

        # We sample in topological order
        for i in self.topo_order:
            if i == index:
                # Intervene: force Y_i = value
                y[:, i] = value
            else:
                ctx_i = self._build_ctx_i(y, i, x_ext=x_ctx)
                flow_i = self.node_flows[i](ctx_i)
                if ctx_i is None:
                    u_i = flow_i.base.rsample((N,))
                else:
                    u_i = flow_i.base.rsample()
                u_i = u_i.reshape(N, -1)
                y_i = flow_i.transform.inv(u_i)
                y_i = y_i.reshape(N, -1)
                y[:, i] = y_i[:, 0]

        if scaler is not None:
            # inverse of scaling if you scaled inputs when training
            y = scaler.inv(y)

        return y

    @torch.no_grad()
    def compute_counterfactual(
        self,
        y_factual: Tensor,
        index: int,
        value: float,
        x_ctx: Tensor = None,
        scaler=None,
        return_dict: bool = False,
    ) -> Tensor:
        """
        Counterfactual: given factual y_factual, apply do(Y_index = value).

        Algorithm:
        - Abduction: compute exogenous u_i for each node from y_factual and factual parents.
        - Action: change the structural equation of node index to constant value (u_index no longer used).
        - Prediction: propagate exogenous u_i and new parents through CTMs in topological order.
        """
        output = {}
        if scaler is not None:
            y = scaler(y_factual)
        else:
            y = y_factual

        N, d = y.shape
        assert d == len(self.node_flows)

        # 1) Abduction: infer exogenous latents u_i under factual context
        u_list = []
        for i in range(d):
            ctx_i = self._build_ctx_i(y, i, x_ext=x_ctx)
            flow_i = self.node_flows[i](ctx_i)
            y_i = y[:, i : i + 1]
            # transform(y_i) = u_i
            u_i = flow_i.transform(y_i)
            u_list.append(u_i)  # list of (N, 1)

        # 2) Action + 3) Prediction: build counterfactual y_cf
        y_cf = torch.zeros_like(y)
        for i in self.topo_order:
            if i == index:
                # do(Y_index = value)
                y_cf[:, i] = value
            else:
                # new context from already-updated parents
                ctx_i_cf = self._build_ctx_i(y_cf, i, x_ext=x_ctx)
                flow_i_cf = self.node_flows[i](ctx_i_cf)
                # keep exogenous u_i same as factual
                u_i = u_list[i]
                y_i_cf = flow_i_cf.transform.inv(u_i)
                y_i_cf = y_i_cf.reshape(N, -1)
                y_cf[:, i] = y_i_cf[:, 0]

        if scaler is not None:
            y_cf = scaler.inv(y_cf)

        if return_dict:
            output["y_cf"] = y_cf
            output["u_factual"] = torch.cat(u_list, dim=-1)
            return output
        else:
            return y_cf

    @torch.no_grad()
    def do_x(
        self,
        y_factual: Tensor,
        x_factual: Tensor,
        x_tilde: Tensor,
        scaler=None,
    ) -> Tensor:
        """
        Optional: intervention on external context X (if you use it).

        Here we treat x_ctx as exogenous causes of Y but not part of the DAG itself.
        We:
            - Abduct u_i under factual x_factual,
            - Predict Y under new context x_tilde using the same u_i.
        """
        if self.context_dim_ext == 0:
            raise RuntimeError("do_x called but context_dim_ext == 0.")

        if scaler is not None:
            y = scaler(y_factual)
        else:
            y = y_factual

        N, d = y.shape
        assert d == len(self.node_flows)

        # Abduction: u_i under factual x_factual
        u_list = []
        for i in range(d):
            ctx_i = self._build_ctx_i(y, i, x_ext=x_factual)
            flow_i = self.node_flows[i](ctx_i)
            y_i = y[:, i : i + 1]
            u_i = flow_i.transform(y_i)
            u_list.append(u_i)

        # Prediction under new context x_tilde, same u_i
        y_cf = torch.zeros_like(y)
        for i in self.topo_order:
            ctx_i_tilde = self._build_ctx_i(y_cf, i, x_ext=x_tilde)
            flow_i_tilde = self.node_flows[i](ctx_i_tilde)
            y_i_cf = flow_i_tilde.transform.inv(u_list[i])
            y_i_cf = y_i_cf.reshape(N, -1)
            y_cf[:, i] = y_i_cf[:, 0]

        if scaler is not None:
            y_cf = scaler.inv(y_cf)

        return y_cf
