"""
Example usage of CausalCTMLightning with node-wise and joint-flow CausalCTM.

This demonstrates how to switch between node-wise and joint-flow CTM approaches
using a simple config flag.
"""

import torch
import zuko.flows
from causal_nf.models.causal_ctm import CausalCTMLightning
from causal_nf.modules.causal_ctm_nodewise_builder import build_nodewise_ctm_flows
from causal_nf.modules.causal_ctm_nodewise import CausalCTM as CausalCTMNodewise
from causal_nf.modules.causal_ctm_joint import CausalCTMJoint


# ============================================================================
# Example 1: Automatic node-wise CTM with default hyperparameters
# ============================================================================
def example1_auto_nodewise_default(preparator):
    """
    Simplest usage: let CausalCTMLightning automatically build a node-wise CTM
    with default NSF flows and default hyperparameters.
    """
    model = CausalCTMLightning(
        preparator=preparator,
        context_dim=0,              # no external context
        use_nodewise=True,          # build node-wise CTM
        # flow_cls and flow_kwargs will use defaults
    )
    return model


# ============================================================================
# Example 2: Automatic node-wise CTM with custom hyperparameters
# ============================================================================
def example2_auto_nodewise_custom(preparator, ctx_dim_ext=0):
    """
    Let CausalCTMLightning automatically build a node-wise CTM with
    custom flow class and hyperparameters.
    """
    model = CausalCTMLightning(
        preparator=preparator,
        context_dim=ctx_dim_ext,
        use_nodewise=True,
        flow_cls=zuko.flows.NSF,     # strictly monotone rational-quadratic splines
        flow_kwargs=dict(
            bins=8,
            transforms=3,             # number of autoregressive transforms
            hidden_features=(64, 64),
        ),
    )
    return model


# ============================================================================
# Example 3: Manual node-wise CTM construction
# ============================================================================
def example3_manual_nodewise(preparator, ctx_dim_ext=0):
    """
    Manually build the node-wise CTM and pass it to CausalCTMLightning.
    This gives you full control over the model construction.
    """
    # Get adjacency matrix
    adj = preparator.adjacency()
    
    # Build one CTM-style flow per node
    node_flows = build_nodewise_ctm_flows(
        adjacency=adj,
        context_dim_ext=ctx_dim_ext,
        flow_cls=zuko.flows.NSF,
        flow_kwargs=dict(
            bins=8,
            transforms=3,
            hidden_features=(64, 64),
        ),
    )
    
    # Create the node-wise CausalCTM
    causal_ctm = CausalCTMNodewise(
        node_flows=node_flows,
        context_dim_ext=ctx_dim_ext,
    )
    causal_ctm.set_adjacency(adj)
    
    # Pass to Lightning module
    model = CausalCTMLightning(
        preparator=preparator,
        model=causal_ctm,            # pre-built model
        context_dim=ctx_dim_ext,
    )
    return model


# ============================================================================
# Example 4: Manual joint-flow CTM construction
# ============================================================================
def example4_manual_joint_flow(preparator, ctx_dim=0):
    """
    Manually build a joint-flow CTM and pass it to CausalCTMLightning.
    
    For joint-flow CTM, you need to build a flow that models all d dimensions
    jointly (not node-wise).
    """
    # Get adjacency matrix
    adj = preparator.adjacency()
    d = adj.shape[0]
    
    # Build a joint flow for all d variables
    # Example: use zuko's MAF (Masked Autoregressive Flow)
    joint_flow = zuko.flows.MAF(
        features=d,
        context=ctx_dim,
        transforms=5,
        hidden_features=(128, 128),
    )
    
    # Create the joint-flow CausalCTM
    causal_ctm = CausalCTMJoint(
        flow=joint_flow,
        context_dim=ctx_dim,
    )
    causal_ctm.set_adjacency(adj)
    
    # Pass to Lightning module
    model = CausalCTMLightning(
        preparator=preparator,
        model=causal_ctm,            # pre-built model
        context_dim=ctx_dim,
    )
    return model


# ============================================================================
# Example 5: Switching between node-wise and joint-flow via config
# ============================================================================
def example5_config_based(preparator, config):
    """
    Switch between node-wise and joint-flow based on a config dict.
    This is useful for hyperparameter tuning and experiments.
    """
    use_nodewise = config.get("use_nodewise", True)
    ctx_dim = config.get("context_dim", 0)
    
    if use_nodewise:
        # Automatic node-wise construction
        model = CausalCTMLightning(
            preparator=preparator,
            context_dim=ctx_dim,
            use_nodewise=True,
            flow_cls=zuko.flows.NSF,
            flow_kwargs=config.get("flow_kwargs", {
                "bins": 8,
                "transforms": 3,
                "hidden_features": (64, 64),
            }),
        )
    else:
        # For joint-flow, you need to build manually
        adj = preparator.adjacency()
        d = adj.shape[0]
        joint_flow = zuko.flows.MAF(
            features=d,
            context=ctx_dim,
            **config.get("flow_kwargs", {
                "transforms": 5,
                "hidden_features": (128, 128),
            })
        )
        causal_ctm = CausalCTMJoint(flow=joint_flow, context_dim=ctx_dim)
        causal_ctm.set_adjacency(adj)
        
        model = CausalCTMLightning(
            preparator=preparator,
            model=causal_ctm,
            context_dim=ctx_dim,
        )
    
    return model


# ============================================================================
# Example 6: With external covariates as context
# ============================================================================
def example6_with_external_context(preparator, ctx_dim_ext=5):
    """
    Use external covariates as context for conditioning.
    """
    adj = preparator.adjacency()
    
    # Build node-wise flows with external context
    node_flows = build_nodewise_ctm_flows(
        adjacency=adj,
        context_dim_ext=ctx_dim_ext,  # e.g., 5 exogenous covariates
        flow_cls=zuko.flows.NSF,
        flow_kwargs=dict(
            bins=8,
            transforms=3,
            hidden_features=(64, 64),
        ),
    )
    
    causal_ctm = CausalCTMNodewise(
        node_flows=node_flows,
        context_dim_ext=ctx_dim_ext,
    )
    causal_ctm.set_adjacency(adj)
    
    model = CausalCTMLightning(
        preparator=preparator,
        model=causal_ctm,
        context_dim=ctx_dim_ext,  # must match
    )
    return model


# ============================================================================
# Usage in training
# ============================================================================
if __name__ == "__main__":
    # Assuming you have a preparator set up
    # preparator = YourPreparator(...)
    
    # Config-based approach (most flexible)
    config = {
        "use_nodewise": True,        # Toggle this to switch approaches!
        "context_dim": 0,
        "flow_kwargs": {
            "bins": 8,
            "transforms": 3,
            "hidden_features": (64, 64),
        }
    }
    
    # model = example5_config_based(preparator, config)
    
    # Then use with PyTorch Lightning Trainer:
    # trainer = pl.Trainer(...)
    # trainer.fit(model, train_dataloader, val_dataloader)
    
    print("CausalCTMLightning usage examples defined successfully!")
    print("\nKey features:")
    print("- Automatic node-wise CTM construction with use_nodewise=True")
    print("- Manual model construction for full control")
    print("- Support for external context/covariates")
    print("- Easy switching between node-wise and joint-flow via config flag")

