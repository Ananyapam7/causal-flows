import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict, Optional

import zuko  # make sure zuko is installed


def build_nodewise_ctm_flows(
    adjacency: Tensor,
    context_dim_ext: int = 0,
    flow_cls: Callable = zuko.flows.NSF,
    flow_kwargs: Optional[Dict] = None,
) -> nn.ModuleList:
    """
    Build one 1D conditional flow per node, to be used in a node-wise CausalCTM.

    Args
    ----
    adjacency:
        Tensor of shape (d, d). adjacency[i, j] != 0 means edge i -> j.
        Must represent a DAG (no cycles).

    context_dim_ext:
        External context dimension (e.g. exogenous covariates not in the DAG).
        For each node i, the effective context dimension will be:
            |pa(i)| + context_dim_ext.

    flow_cls:
        A zuko flow class, e.g. zuko.flows.NSF or zuko.flows.MAF.
        Must have signature roughly:
            flow_cls(features: int, context: int, **flow_kwargs)
        and support:
            flow_i(ctx).log_prob(y_i)
            flow_i(ctx).base.rsample(shape)
            flow_i(ctx).transform / .transform.inv

    flow_kwargs:
        Extra keyword arguments passed to `flow_cls` for every node.
        For example:
            flow_kwargs = dict(transforms=3, hidden_features=(64, 64))

    Returns
    -------
    node_flows:
        nn.ModuleList of length d. node_flows[i] is a conditional flow for node i:
            flow_i(ctx_i) ~ p(y_i | parents_i, external context).
    """
    if not isinstance(adjacency, torch.Tensor):
        adjacency = torch.as_tensor(adjacency)

    adjacency = adjacency.detach()
    d = adjacency.shape[0]
    assert adjacency.shape[0] == adjacency.shape[1], "Adjacency must be square."

    flow_kwargs = {} if flow_kwargs is None else dict(flow_kwargs)

    node_flows = nn.ModuleList()

    for i in range(d):
        # Parents of node i are those j with edge j -> i
        pa_i = torch.where(adjacency[:, i] != 0)[0]
        num_parents = int(pa_i.numel())

        # Context dimension for node i: parents + external context
        ctx_dim_i = num_parents + int(context_dim_ext)

        # Build kwargs per node
        flow_kwargs_i = dict(flow_kwargs)

        # Ensure adjacency has at least self-dependency to avoid null Jacobians
        if "adjacency" not in flow_kwargs_i:
            adjacency_i = torch.ones(1, 1 + ctx_dim_i, dtype=torch.bool)
            flow_kwargs_i["adjacency"] = adjacency_i

        # zuko flows expect an int context dimension; 0 is fine for no context
        flow_i = flow_cls(
            features=1,
            context=ctx_dim_i,
            **flow_kwargs_i,
        )

        node_flows.append(flow_i)

    return node_flows
