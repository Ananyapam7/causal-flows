"""Comparison test script for CausalNF and CausalCTMJoint models

This script tests both CausalNormalizingFlow and CausalCTMJoint side by side
to demonstrate their similarities and differences.
"""
import torch
import zuko.flows as zflows

# Import both models using direct import to avoid GNN dependency issues
import importlib.util
import sys

# Direct import of CTM to bypass __init__.py
spec_ctm = importlib.util.spec_from_file_location(
    "ctm", "causal_nf/modules/causal_ctm_joint.py"
)
ctm_module = importlib.util.module_from_spec(spec_ctm)
sys.modules["ctm_module"] = ctm_module
spec_ctm.loader.exec_module(ctm_module)
CausalCTMJoint = ctm_module.CausalCTMJoint

# Direct import of CausalNF to bypass __init__.py
spec_nf = importlib.util.spec_from_file_location(
    "causal_nf", "causal_nf/modules/causal_nf.py"
)
nf_module = importlib.util.module_from_spec(spec_nf)
sys.modules["nf_module"] = nf_module
spec_nf.loader.exec_module(nf_module)
CausalNormalizingFlow = nf_module.CausalNormalizingFlow

print("=" * 80)
print("COMPARISON TEST: CausalNormalizingFlow vs CausalCTMJoint")
print("=" * 80)

# ============================================================================
# 1. INITIALIZATION COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("1. INITIALIZATION")
print("=" * 80)

# Create flows for both models
print("\nCreating MAF flows...")
flow_nf = zflows.MAF(
    features=3,
    context=0,
    transforms=2,
    hidden_features=[32, 32],
    base_to_data=False,
    base_distr="normal",
    learn_base=True,
)

flow_ctm = zflows.MAF(
    features=3,
    context=0,
    transforms=2,
    hidden_features=[32, 32],
    base_to_data=False,
    base_distr="normal",
    learn_base=True,
)

print("\n--- CausalNormalizingFlow ---")
print("Initialization: CausalNormalizingFlow(flow=flow)")
print("Parameters: flow only")
causal_nf = CausalNormalizingFlow(flow=flow_nf)
adj = torch.eye(3)
causal_nf.set_adjacency(adj)
print("  [OK] Initialized successfully")

print("\n--- CausalCTMJoint ---")
print("Initialization: CausalCTMJoint(flow=flow, context_dim=0)")
print("Parameters: flow + context_dim")
causal_ctm = CausalCTMJoint(flow=flow_ctm, context_dim=0)
causal_ctm.set_adjacency(adj)
print("  [OK] Initialized successfully")

# ============================================================================
# 2. FORWARD PASS COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("2. FORWARD PASS")
print("=" * 80)

x_nf = torch.randn(5, 3)  # NF uses 'x'
y_ctm = torch.randn(5, 3)  # CTM uses 'y'

print("\n--- CausalNormalizingFlow ---")
print(f"Input: x with shape {x_nf.shape}")
print("Method: forward(x)")
print("Flow usage: self.flow() - unconditional flow")
output_nf = causal_nf.forward(x_nf)
print(f"  log_prob shape: {output_nf['log_prob'].shape}")
print(f"  log_prob mean: {output_nf['log_prob'].mean().item():.4f}")
print(f"  loss shape: {output_nf['loss'].shape}")

print("\n--- CausalCTMJoint ---")
print(f"Input: y with shape {y_ctm.shape}")
print("Method: forward(y, x_ctx=None)")
print("Flow usage: self.flow(x_ctx) - conditional flow")
output_ctm = causal_ctm.forward(y_ctm)
print(f"  log_prob shape: {output_ctm['log_prob'].shape}")
print(f"  log_prob mean: {output_ctm['log_prob'].mean().item():.4f}")
print(f"  loss shape: {output_ctm['loss'].shape}")

# ============================================================================
# 3. SAMPLING COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("3. SAMPLING")
print("=" * 80)

print("\n--- CausalNormalizingFlow ---")
print("Method: sample(shape=(10,))")
print("Returns: {'x_obs', 'u_obs'}")
samples_nf = causal_nf.sample(shape=(10,))
print(f"  x_obs shape: {samples_nf['x_obs'].shape}")
print(f"  u_obs shape: {samples_nf['u_obs'].shape}")

print("\n--- CausalCTMJoint ---")
print("Method: sample_conditional(x_ctx, shape=(10,))")
print("Returns: {'y_obs', 'u_obs'}")
print("Note: Requires context (can be None if context_dim=0)")
samples_ctm = causal_ctm.sample_conditional(None, shape=(10,))
print(f"  y_obs shape: {samples_ctm['y_obs'].shape}")
print(f"  u_obs shape: {samples_ctm['u_obs'].shape}")

# ============================================================================
# 4. LOG PROBABILITY COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("4. LOG PROBABILITY")
print("=" * 80)

print("\n--- CausalNormalizingFlow ---")
print("Method: log_prob(x, scaler=None)")
log_prob_nf = causal_nf.log_prob(x_nf)
print(f"  log_prob shape: {log_prob_nf.shape}")
print(f"  log_prob mean: {log_prob_nf.mean().item():.4f}")

print("\n--- CausalCTMJoint ---")
print("Method: log_prob(y, x_ctx=None, scaler=None)")
log_prob_ctm = causal_ctm.log_prob(y_ctm)
print(f"  log_prob shape: {log_prob_ctm.shape}")
print(f"  log_prob mean: {log_prob_ctm.mean().item():.4f}")

# ============================================================================
# 5. INTERVENTION COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("5. INTERVENTION")
print("=" * 80)

print("\n--- CausalNormalizingFlow ---")
print("Method: intervene(index=0, value=1.0, shape=(5,), scaler=None)")
x_int_nf = causal_nf.intervene(index=0, value=1.0, shape=(5,))
print(f"  Output shape: {x_int_nf.shape}")
print(f"  Output mean: {x_int_nf.mean(0)}")

print("\n--- CausalCTMJoint ---")
print("Method: intervene(index=0, value=1.0, x_ctx=None, shape=(5,), scaler=None)")
y_int_ctm = causal_ctm.intervene(index=0, value=1.0, shape=(5,))
print(f"  Output shape: {y_int_ctm.shape}")
print(f"  Output mean: {y_int_ctm.mean(0)}")

# ============================================================================
# 6. COUNTERFACTUAL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("6. COUNTERFACTUAL")
print("=" * 80)

print("\n--- CausalNormalizingFlow ---")
print("Method: compute_counterfactual(x_factual, index=0, value=1.0, scaler=None)")
x_cf_nf = causal_nf.compute_counterfactual(x_factual=x_nf, index=0, value=1.0)
print(f"  Output shape: {x_cf_nf.shape}")
print(f"  Output mean: {x_cf_nf.mean(0)}")

print("\n--- CausalCTMJoint ---")
print("Method: compute_counterfactual(y_factual, index=0, value=1.0, x_ctx=None, scaler=None)")
y_cf_ctm = causal_ctm.compute_counterfactual(y_factual=y_ctm, index=0, value=1.0)
print(f"  Output shape: {y_cf_ctm.shape}")
print(f"  Output mean: {y_cf_ctm.mean(0)}")

# ============================================================================
# 7. KEY DIFFERENCES SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("7. KEY DIFFERENCES SUMMARY")
print("=" * 80)

differences = [
    ("Initialization",
     "CausalNormalizingFlow(flow)",
     "CausalCTMJoint(flow, context_dim=0)"),
    ("Variable naming", 
     "Uses 'x' for variables", 
     "Uses 'y' for output variables"),
    ("Flow usage",
     "Unconditional: self.flow()",
     "Conditional: self.flow(x_ctx)"),
    ("Forward signature", 
     "forward(x)", 
     "forward(y, x_ctx=None)"),
    ("Sampling",
     "sample(shape)",
     "sample_conditional(x_ctx, shape)"),
    ("Context support", 
     "No context support", 
     "Supports context variables (x_ctx)"),
    ("Additional methods", 
     "Has vi(), compute_ate(), compute_jacobian()", 
     "Has do_x() for context interventions"),
]

print("\n{:<20} {:<35} {:<35}".format("Feature", "CausalNF", "CausalCTM"))
print("-" * 90)
for feature, nf_val, ctm_val in differences:
    print(f"{feature:<20} {nf_val:<35} {ctm_val:<35}")

# ============================================================================
# 8. FUNCTIONALITY COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("8. FUNCTIONALITY COMPARISON")
print("=" * 80)

print("\nCommon methods:")
common_methods = [
    "set_adjacency()",
    "forward()",
    "log_prob()",
    "intervene()",
    "compute_counterfactual()",
]
for method in common_methods:
    print(f"  [OK] {method}")

print("\nCausalNF-only methods:")
nf_only = [
    "vi() - Variational Inference",
    "sample() - Unconditional sampling",
    "compute_ate() - Average Treatment Effect",
    "compute_jacobian() - Jacobian computation",
    "_inverse() - Inverse transform",
]
for method in nf_only:
    print(f"  - {method}")

print("\nCausalCTM-only methods:")
ctm_only = [
    "sample_conditional() - Conditional sampling",
    "do_x() - Context variable intervention",
    "_ensure_ctx() - Context handling helper",
]
for method in ctm_only:
    print(f"  - {method}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE!")
print("=" * 80)
print("\nBoth models work correctly. The main difference is that CTM supports")
print("context variables (conditional modeling) while CausalNF is unconditional.")
