"""
Quick Start: How to Wire Up Node-wise CausalCTM
=================================================

This script shows exactly how to wire up the node-wise CausalCTM with
CausalCTMLightning, as requested.
"""

import torch
import zuko.flows
from causal_nf.models.causal_ctm import CausalCTMLightning
from causal_nf.modules.causal_ctm_nodewise_builder import build_nodewise_ctm_flows
from causal_nf.modules.causal_ctm_nodewise import CausalCTM


# ============================================================================
# Method 1: Manual Wiring (Full Control)
# ============================================================================

def manual_wiring(preparator):
    """
    Explicitly build the node-wise CTM and wire it to CausalCTMLightning.
    This gives you full control over the model construction.
    """
    # adjacency: (d, d) tensor for variables y_0,...,y_{d-1}
    adj = preparator.adjacency()  # or however you get it
    
    # Build one CTM-style flow per node (using NSF here)
    node_flows = build_nodewise_ctm_flows(
        adjacency=adj,
        context_dim_ext=0,                    # or >0 if you also condition on x_ctx
        flow_cls=zuko.flows.NSF,             # strictly monotone rational-quadratic splines
        flow_kwargs=dict(
            bins=8,
            transforms=3,                    # number of autoregressive transforms
            hidden_features=(64, 64),
        ),
    )
    
    model = CausalCTM(
        node_flows=node_flows,
        context_dim_ext=0,                    # must match what you passed above
    )
    model.set_adjacency(adj)
    
    # Wrap in Lightning module
    lightning_model = CausalCTMLightning(
        preparator=preparator,
        model=model,
        context_dim=0,
    )
    
    return lightning_model


# ============================================================================
# Method 2: Automatic Wiring (Simplest)
# ============================================================================

def automatic_wiring(preparator):
    """
    Let CausalCTMLightning build everything automatically.
    Just pass the flow config and it handles the rest!
    """
    lightning_model = CausalCTMLightning(
        preparator=preparator,
        context_dim=0,
        use_nodewise=True,                   # Toggle to enable node-wise CTM
        flow_cls=zuko.flows.NSF,            # Flow class
        flow_kwargs=dict(
            bins=8,
            transforms=3,
            hidden_features=(64, 64),
        ),
    )
    
    return lightning_model


# ============================================================================
# With External Covariates
# ============================================================================

def with_external_context(preparator, ctx_dim_ext=5):
    """
    If you later decide to add external covariates as context.
    """
    # Method 1: Manual
    adj = preparator.adjacency()
    
    node_flows = build_nodewise_ctm_flows(
        adjacency=adj,
        context_dim_ext=ctx_dim_ext,          # e.g. number of exogenous covariates
        flow_cls=zuko.flows.NSF,
        flow_kwargs=dict(bins=8, transforms=3, hidden_features=(64, 64)),
    )
    
    model = CausalCTM(
        node_flows=node_flows,
        context_dim_ext=ctx_dim_ext,
    )
    model.set_adjacency(adj)
    
    lightning_model = CausalCTMLightning(
        preparator=preparator,
        model=model,
        context_dim=ctx_dim_ext,
    )
    
    # OR Method 2: Automatic
    # lightning_model = CausalCTMLightning(
    #     preparator=preparator,
    #     context_dim=ctx_dim_ext,
    #     use_nodewise=True,
    #     flow_cls=zuko.flows.NSF,
    #     flow_kwargs=dict(bins=8, transforms=3, hidden_features=(64, 64)),
    # )
    
    return lightning_model


# ============================================================================
# Using the Model
# ============================================================================

def usage_example(preparator, train_loader, val_loader):
    """
    Complete example: build, train, and use the model.
    """
    import pytorch_lightning as pl
    
    # Build model
    model = automatic_wiring(preparator)
    
    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # From that point on:
    # - model.forward(y, x_ctx) / model.log_prob(...) uses one CTM per node
    # - sample_conditional, intervene, and compute_counterfactual follow the
    #   SCM semantics with genuine node-wise conditionals
    
    # Example: Interventional sampling
    with torch.no_grad():
        y_int = model.model.intervene(
            index=0,              # intervene on first variable
            value=1.0,            # set to 1.0
            shape=(100,),         # sample 100 examples
            scaler=preparator.scaler_transform,
        )
        print(f"Interventional samples shape: {y_int.shape}")
    
    # Example: Counterfactual inference
    with torch.no_grad():
        y_factual = next(iter(val_loader))[0][:10]  # get 10 examples
        y_cf = model.model.compute_counterfactual(
            y_factual=y_factual,
            index=1,              # intervene on second variable
            value=0.5,            # set to 0.5
            scaler=preparator.scaler_transform,
        )
        print(f"Counterfactual samples shape: {y_cf.shape}")


# ============================================================================
# Config-based Approach for Experiments
# ============================================================================

def config_driven_wiring(preparator, config):
    """
    Build model from config dict - useful for hyperparameter sweeps.
    """
    if config.get("use_nodewise", True):
        # Node-wise CTM (recommended)
        model = CausalCTMLightning(
            preparator=preparator,
            context_dim=config.get("context_dim", 0),
            use_nodewise=True,
            flow_cls=zuko.flows.NSF,  # or get from config
            flow_kwargs=config.get("flow_kwargs", {
                "bins": 8,
                "transforms": 3,
                "hidden_features": (64, 64),
            }),
        )
    else:
        # Joint-flow CTM (for comparison)
        # You'd need to build the joint flow manually here
        raise NotImplementedError("See examples/ctm_usage_example.py")
    
    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Node-wise CausalCTM Quick Start")
    print("=" * 70)
    print()
    print("Three ways to wire up node-wise CausalCTM:")
    print()
    print("1. Manual wiring (full control):")
    print("   - Build node_flows with build_nodewise_ctm_flows()")
    print("   - Create CausalCTM(node_flows=...)")
    print("   - Wrap in CausalCTMLightning(model=...)")
    print()
    print("2. Automatic wiring (simplest):")
    print("   - Just use CausalCTMLightning(use_nodewise=True, flow_cls=..., flow_kwargs=...)")
    print("   - Everything built automatically!")
    print()
    print("3. Config-driven (experiments):")
    print("   - Store settings in config dict")
    print("   - Easy to switch between approaches")
    print("   - Perfect for hyperparameter tuning")
    print()
    print("See function definitions above for complete code examples.")
    print()
    print("Key advantage: Just toggle use_nodewise=True/False to switch")
    print("between node-wise and joint-flow CausalCTM!")

