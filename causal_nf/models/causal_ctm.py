"""
CausalCTM Lightning Module

This module contains the PyTorch Lightning wrapper for Causal Continuous
Transformation Models (CTM), supporting both joint-flow and node-wise approaches.
"""

import os
import time

import matplotlib.pyplot as plt
import torch
import wandb
from tueplots import bundles

import causal_nf.utils.io as causal_io
from causal_nf.models.base_model import BaseLightning
from causal_nf.utils.optimizers import build_optimizer, build_scheduler

plt.rcParams.update(bundles.icml2022())
from causal_nf.utils.pairwise.mmd import maximum_mean_discrepancy
from causal_nf.modules.causal_ctm_nodewise import CausalCTM as CausalCTMNodewise
from causal_nf.modules.causal_ctm_joint import CausalCTMJoint
from causal_nf.modules.causal_ctm_nodewise_builder import (
    build_nodewise_ctm_flows,
)

import numpy as np


class CausalCTMLightning(BaseLightning):
    """
    PyTorch Lightning module for Causal Continuous Transformation Models.
    
    Supports both:
    - Node-wise CTM: One conditional flow per node (recommended)
    - Joint-flow CTM: Single flow for all variables (legacy)
    """
    
    def __init__(
        self,
        preparator,
        model=None,
        context_dim: int = 0,
        init_fn=None,
        plot=True,
        regularize=False,
        # Config for automatic model building
        use_nodewise: bool = True,
        flow_cls=None,
        flow_kwargs=None,
        joint_flow_cls=None,
        joint_flow_kwargs=None,
    ):
        """
        Initialize CausalCTMLightning module.
        
        Args:
            preparator: Data preparator with adjacency matrix and data info.
            model: Optional pre-built CausalCTM model (joint or node-wise).
                   If None, will build automatically based on use_nodewise flag.
            context_dim: External context dimension.
            init_fn: Optional initialization function.
            plot: Whether to generate plots during validation/test.
            regularize: Whether to regularize Jacobian with adjacency constraints
                       (only works with joint-flow CTM).
            use_nodewise: If True and model is None, build node-wise CTM.
                          If False and model is None, build joint-flow CTM.
            flow_cls: Flow class for building node-wise CTMs (e.g., zuko.flows.NSF).
                      Only used if model is None and use_nodewise is True.
            flow_kwargs: Dict of kwargs for node-wise flow_cls (e.g., bins, transforms,
                         hidden_features). Only used if model is None and use_nodewise is True.
            joint_flow_cls: Flow class for joint CTM construction (defaults to zuko.flows.MAF).
            joint_flow_kwargs: Dict of kwargs for the joint flow class (e.g., transforms,
                               hidden_features). Only used if model is None and use_nodewise is False.
        """
        super(CausalCTMLightning, self).__init__(preparator, init_fn=init_fn)

        self.plot = plot
        self.regularize = regularize
        self.context_dim = int(context_dim)
        self.use_nodewise = use_nodewise

        # Build model if not provided
        if model is None:
            model = self._build_model(
                preparator=preparator,
                use_nodewise=use_nodewise,
                context_dim=context_dim,
                flow_cls=flow_cls,
                flow_kwargs=flow_kwargs,
                joint_flow_cls=joint_flow_cls,
                joint_flow_kwargs=joint_flow_kwargs,
            )
        
        self.model = model
        
        self.set_input_scaler()
        self.reset_parameters()

    @staticmethod
    def _build_model(
        preparator,
        use_nodewise: bool,
        context_dim: int,
        flow_cls=None,
        flow_kwargs=None,
        joint_flow_cls=None,
        joint_flow_kwargs=None,
    ):
        """
        Build a CausalCTM model automatically based on config.
        
        Args:
            preparator: Data preparator (needed for adjacency matrix).
            use_nodewise: If True, build node-wise CTM; otherwise build joint-flow CTM.
            context_dim: External context dimension.
            flow_cls: Flow class for node-wise CTM (e.g., zuko.flows.NSF).
            flow_kwargs: Flow kwargs dict for node-wise CTM.
            joint_flow_cls: Flow class for joint CTM (defaults to zuko.flows.MAF).
            joint_flow_kwargs: Flow kwargs dict for joint CTM.
            
        Returns:
            CausalCTM model (either node-wise or joint-flow).
        """
        adj = preparator.adjacency()
        
        if use_nodewise:
            # Build node-wise CausalCTM
            if flow_cls is None:
                # Default to NSF if not specified
                import zuko.flows
                flow_cls = zuko.flows.NSF
            
            if flow_kwargs is None:
                # Default hyperparameters
                flow_kwargs = dict(
                    bins=8,
                    transforms=3,
                    hidden_features=(64, 64),
                )
            
            node_flows = build_nodewise_ctm_flows(
                adjacency=adj,
                context_dim_ext=context_dim,
                flow_cls=flow_cls,
                flow_kwargs=flow_kwargs,
            )
            
            model = CausalCTMNodewise(
                node_flows=node_flows,
                context_dim_ext=context_dim,
            )
        else:
            # Build joint-flow CausalCTM
            import zuko.flows as zflows

            d = adj.shape[0]
            if joint_flow_cls is None:
                joint_flow_cls = zflows.MAF
            joint_kwargs = dict(
                transforms=3,
                hidden_features=(64, 64),
            )
            if joint_flow_kwargs is not None:
                joint_kwargs.update(joint_flow_kwargs)
            if isinstance(joint_kwargs.get("hidden_features"), list):
                joint_kwargs["hidden_features"] = tuple(joint_kwargs["hidden_features"])

            flow = joint_flow_cls(
                features=d,
                context=context_dim,
                **joint_kwargs,
            )
            model = CausalCTMJoint(flow=flow, context_dim=context_dim)
        
        model.set_adjacency(adj)
        return model

    def reset_parameters(self):
        super(CausalCTMLightning, self).reset_parameters()

    def set_input_scaler(self):
        """Set input scaler and adjacency matrix."""
        self.input_scaler = self.preparator.get_scaler(fit=True)
        print(self.input_scaler)
        self.model.set_adjacency(self.preparator.adjacency())

    def _split_batch(self, batch):
        """Split batch into y and optional external context."""
        y = batch[0].to(self.device)
        x_ctx = None
        if len(batch) > 1 and self.context_dim > 0:
            cand = batch[1].to(self.device)
            if cand.dim() == 2 and cand.shape[-1] == self.context_dim:
                x_ctx = cand
        return y, x_ctx

    def get_y_norm(self, y):
        """Normalize y using input scaler."""
        return self.input_scaler.transform(y, inplace=False)

    def forward(self, batch, **kwargs):
        """Forward pass: compute log probability and loss."""
        y, x_ctx = self._split_batch(batch)
        y_norm = self.get_y_norm(y)

        tic = time.time()
        output = self.model(y_norm, x_ctx)
        
        if self.regularize and isinstance(self.model, CausalCTMJoint):
            # Compute Jacobian of T_x at mean(y) - only for joint-flow CTM
            n_flow = self.model.flow(self.model._ensure_ctx(y_norm, x_ctx))
            jac = torch.autograd.functional.jacobian(
                n_flow.transform, y_norm.mean(0), create_graph=True
            )
            adj = self.preparator.adjacency(True)
            loss_ = torch.norm(jac[(adj == 0.0)], p=2)
            output["loss"] = output["loss"] + loss_
        
        output["time_forward"] = self.compute_time(tic, y_norm.shape[0])
        return output

    def compute_time(self, tic, num_samples):
        """Compute time per sample in microseconds."""
        delta_time = (time.time() - tic) * 1000
        return torch.tensor(delta_time / num_samples * 1000)

    @torch.no_grad()
    def predict(
        self,
        batch,
        observational=False,
        intervene=False,
        counterfactual=False,
        do_x=False,
    ):
        """
        Prediction step with optional causal queries.
        
        Args:
            batch: Input batch
            observational: Whether to sample from observational distribution
            intervene: Whether to perform interventions
            counterfactual: Whether to compute counterfactuals
            do_x: Whether to perform interventions on external context
        """
        output = {}
        y, x_ctx = self._split_batch(batch)
        n = y.shape[0]

        # Compute log probability
        tic = time.time()
        log_prob = self.model.log_prob(
            y, x_ctx=x_ctx, scaler=self.preparator.scaler_transform
        )
        output["time_log_prob"] = self.compute_time(tic, n)
        output["loss"] = -log_prob
        output["log_prob"] = log_prob

        # Observational sampling
        if observational:
            tic = time.time()
            if isinstance(self.model, CausalCTMJoint):
                x_in = self.model._ensure_ctx(y, x_ctx)
                obs_dict = self.model.sample_conditional(x_in, shape=(n,))
            else:
                obs_dict = self.model.sample_conditional(x_ctx=x_ctx, shape=(n,))
            output["time_sample_obs"] = self.compute_time(tic, n)
            y_obs_norm = obs_dict["y_obs"]
            y_obs = self.input_scaler.inverse_transform(y_obs_norm, inplace=False)
            if self.plot:
                output["y"] = self.preparator.post_process(y)
                output["y_obs"] = self.preparator.post_process(y_obs)
            mmd_value = maximum_mean_discrepancy(y, y_obs, sigma=None)
            output["mmd_obs"] = mmd_value

        # Interventions
        if intervene:
            intervention_list = self.preparator.get_intervention_list()
            delta_times = []
            for int_dict in intervention_list:
                name = int_dict["name"]
                value = int_dict["value"]
                index = int_dict["index"]
                tic = time.time()
                y_int = self.model.intervene(
                    index=index,
                    value=value,
                    x_ctx=x_ctx,
                    shape=(n,),
                    scaler=self.preparator.scaler_transform,
                )
                delta_times.append(self.compute_time(tic, n))
                if self.plot:
                    output[f"y_int_{index + 1}={name}"] = self.preparator.post_process(y_int)

            if len(delta_times) > 0:
                output["time_intervene"] = torch.stack(delta_times).mean()

        # Counterfactuals
        if counterfactual:
            intervention_list = self.preparator.get_intervention_list()
            delta_times = []
            for int_dict in intervention_list:
                name = int_dict["name"]
                value = int_dict["value"]
                index = int_dict["index"]
                tic = time.time()
                y_cf = self.model.compute_counterfactual(
                    y_factual=y,
                    index=index,
                    value=value,
                    x_ctx=x_ctx,
                    scaler=self.preparator.scaler_transform,
                )
                delta_times.append(self.compute_time(tic, n))
                if self.plot:
                    output[f"y_cf_{index + 1}={name}"] = self.preparator.post_process(y_cf)

            if len(delta_times) > 0:
                output["time_cf"] = torch.stack(delta_times).mean()

        # Intervention on external context
        if do_x and self.context_dim > 0 and x_ctx is not None:
            x_tilde = x_ctx[torch.randperm(n)]
            tic = time.time()
            y_cf_x = self.model.do_x(
                y_factual=y,
                x_factual=x_ctx,
                x_tilde=x_tilde,
                scaler=self.preparator.scaler_transform,
            )
            output["time_do_x"] = self.compute_time(tic, n)
            if self.plot:
                output["y_cf_x"] = self.preparator.post_process(y_cf_x)

        return output

    def training_step(self, train_batch, batch_idx):
        """Training step."""
        loss_dict = self(train_batch)
        loss_dict["loss"] = loss_dict["loss"].mean()
        log_dict = {}
        
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            output = self.jacobian_losses(train_batch)
            loss_dict.update(output)

        self.update_log_dict(log_dict=log_dict, my_dict=loss_dict, regex=r"^(?!y_).*$")
        return loss_dict

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        self.eval()

        if self.current_epoch % 10 == 1:
            observational = batch_idx == 0
            intervene = False
        else:
            observational = False
            intervene = False

        loss_dict = self.predict(
            batch,
            observational=observational,
            intervene=intervene,
            counterfactual=False,
            do_x=False,
        )

        log_dict = {}
        self.update_log_dict(
            log_dict=log_dict, my_dict=loss_dict, regex=r"^(?!.*y_).*$"
        )

        if batch_idx == 0 and self.current_epoch % 5 == 0:
            output = self.jacobian_losses(batch)
            log_dict.update(output)

        return log_dict

    def test_step(self, batch, batch_idx):
        """Test step with full causal query evaluation."""
        self.eval()

        observational = batch_idx < 1
        intervene = batch_idx < 1
        counterfactual = batch_idx < 1
        do_x = batch_idx < 1 and self.context_dim > 0

        loss_dict = self.predict(
            batch,
            observational=observational,
            intervene=intervene,
            counterfactual=counterfactual,
            do_x=do_x,
        )

        log_dict = {}
        self.update_log_dict(log_dict=log_dict, my_dict=loss_dict)
        return log_dict

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        self.lr = self.optim_config.base_lr
        causal_io.print_debug(f"Setting lr: {self.lr}")

        params = self.model.parameters()
        opt = build_optimizer(optim_config=self.optim_config, params=params)

        output = {}

        if isinstance(self.optim_config.scheduler, str):
            sched = build_scheduler(optim_config=self.optim_config, optimizer=opt)
            output["optimizer"] = opt
            output["lr_scheduler"] = sched
            output["monitor"] = "val_loss"
        else:
            output["optimizer"] = opt
        return output

    def _plot_jacobian(self, J, title="Jacobian Matrix", variable="y"):
        """Plot Jacobian matrix."""
        if isinstance(J, torch.Tensor):
            J = J.detach().numpy()

        J_abs = np.absolute(J)
        fig, ax = plt.subplots()

        height, width = J.shape
        fig_aspect_ratio = fig.get_figheight() / fig.get_figwidth()
        data_aspect_ratio = (height / width) * fig_aspect_ratio
        cax = ax.matshow(J_abs, aspect=data_aspect_ratio, cmap="viridis")

        fig.colorbar(cax)
        ax.set_title(f"{title} {variable}")

        ax.set_xticks(range(J.shape[1]))
        ax.set_yticks(range(J.shape[0]))

        xticks = [
            "$\\frac{{ \\partial f_m }}{{ \\partial {}_{} }}$".format(variable, i)
            for i in range(1, J.shape[1] + 1)
        ]
        ax.set_xticklabels(xticks)
        yticks = [
            "$\\frac{{ \\partial f_{} }}{{ \\partial {}_n }}$".format(i, variable)
            for i in range(1, J.shape[1] + 1)
        ]
        ax.set_yticklabels(yticks)

        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                value = J[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="w")

        return fig

    def jacobian_losses(self, batch, filename=None):
        """
        Compute Jacobian-based regularization losses.
        
        Note: Only implemented for joint-flow CTM. Node-wise CTM would require
        computing derivatives of each node's conditional flow w.r.t. its parents.
        """
        output = {}
        if isinstance(filename, str):
            plt.close("all")
        
        y, x_ctx = self._split_batch(batch)
        y_norm = self.get_y_norm(y)
        
        # For joint-flow CTM, we compute Jacobian of the flow transform
        if isinstance(self.model, CausalCTMJoint):
            n_flow = self.model.flow(self.model._ensure_ctx(y_norm, x_ctx))
            jac_y = torch.autograd.functional.jacobian(
                lambda y_in: n_flow.transform(y_in), y_norm.mean(0)
            )

            adj = self.preparator.adjacency(True)
            triangular = torch.tril(torch.ones(adj.shape), diagonal=-1).bool()

            if isinstance(filename, str):
                fig = self._plot_jacobian(jac_y, title="Jacobian Matrix", variable="y")
                fig.savefig(f"{filename}y.pdf")

            mask = (adj == 0.0) * triangular
            loss_ = np.absolute(jac_y[mask]).mean()
            output["loss_jacobian_y"] = torch.tensor(loss_)

            if isinstance(filename, str):
                plt.close("all")
        # For node-wise CTM, skip Jacobian computation for now
        
        return output

