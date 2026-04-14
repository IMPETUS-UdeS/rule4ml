"""
Extended wrapper classes for autoresearch experiments.
These subclass TorchModelWrapper to add inference-time post-processing.
"""

import os

import numpy as np
import torch

from rule4ml.models.wrappers import BaseModelWrapper, TorchModelWrapper


class LogTransformWrapper(TorchModelWrapper):
    """
    Inverse-transforms log-space model outputs at inference.

    When training with log1p-transformed targets, the model learns to predict
    log1p(true_value) directly. At inference time we recover:
        pred_raw = expm1(max(pred_log, 0))

    Clamping at 0 before expm1 ensures non-negative predictions (equivalent to
    the model predicting ≤0 in log space → raw prediction of 0). Also rounds
    resource targets (first 4: BRAM/DSP/FF/LUT) to nearest non-negative integer.
    """

    _N_RESOURCE = 4

    def predict_from_df(self, inputs_df, **kwargs):
        preds = super().predict_from_df(inputs_df, **kwargs)
        # Inverse log transform: recover raw values from log1p predictions
        preds = np.expm1(np.maximum(preds, 0.0))
        # Round resource predictions to nearest integer (integer-valued targets)
        preds[:, : self._N_RESOURCE] = np.maximum(
            np.round(preds[:, : self._N_RESOURCE]), 0.0
        )
        return preds


class MCDropoutWrapper(LogTransformWrapper):
    """
    MC-Dropout uncertainty estimation via stochastic inference.

    Runs N_PASSES forward passes with dropout active (keeping BN in eval mode),
    then returns the per-pass variance in log1p space as epistemic uncertainty.

    Key advantage over DER: zero accuracy trade-off — training uses the same
    pure MSE loss as LogTransformWrapper. Uncertainty comes entirely from
    inference-time dropout stochasticity with no gradient to γ.

    Usage: can be applied to any exp28/exp32 checkpoint without retraining.
    For exp33+: consider increasing dropout from 0.1→0.15 for better
    separation of uncertainty across architecture families.
    """

    N_PASSES = 20
    _N_RESOURCE = 4

    def predict_uncertainty_from_df(self, inputs_df):
        """
        Returns (point_estimates, variance) where variance is in log1p space.

        point_estimates: [N, T] in original (non-log) space
        variance: [N, T] variance across MC passes in log1p space
            Higher variance → higher epistemic uncertainty
        """
        passes = []
        for _ in range(self.N_PASSES):
            # Get raw log1p predictions with dropout active; copy df to avoid mutation
            raw = BaseModelWrapper.predict_from_df(
                self, inputs_df.copy(), mc_dropout=True
            )  # [N, T] in log1p space
            passes.append(raw)

        passes = np.stack(passes, axis=0)  # [P, N, T]
        mean_log = np.mean(passes, axis=0)  # [N, T]
        variance = np.var(passes, axis=0)   # [N, T] — epistemic uncertainty in log space

        point_estimates = np.expm1(np.maximum(mean_log, 0.0))
        point_estimates[:, : self._N_RESOURCE] = np.maximum(
            np.round(point_estimates[:, : self._N_RESOURCE]), 0.0
        )
        return point_estimates, variance


class SeparateHeadWrapper(LogTransformWrapper):
    """
    Post-hoc uncertainty heads trained on frozen-backbone residuals.
    """

    _N_RESOURCE = 4

    def __init__(self):
        super().__init__()
        self.uncertainty_calibration = None
        self.uncertainty_spread_weight = None

    @staticmethod
    def _apply_monotonic_calibration(raw_uncertainty, calibration):
        if calibration is None:
            return raw_uncertainty

        calibrated = np.array(raw_uncertainty, copy=True)
        for i, target_cal in enumerate(calibration):
            if target_cal is None:
                continue
            x = np.asarray(target_cal["x"], dtype=np.float32)
            y = np.asarray(target_cal["y"], dtype=np.float32)
            if len(x) == 0 or len(y) == 0:
                continue
            calibrated[:, i] = np.interp(
                calibrated[:, i],
                x,
                y,
                left=float(y[0]),
                right=float(y[-1]),
            )
        return calibrated

    @staticmethod
    def _apply_spread_fusion(mean_uncertainty, std_uncertainty, spread_weight):
        if spread_weight is None or std_uncertainty is None:
            return mean_uncertainty
        fused = np.array(mean_uncertainty, copy=True)
        for i, alpha in enumerate(spread_weight):
            fused[:, i] = fused[:, i] + float(alpha) * std_uncertainty[:, i]
        return fused

    def _save_weights(self, save_dir):
        backbone_path = os.path.join(save_dir, f"{self.model.name}.weights.pt")
        backbone_sd = {
            k: v
            for k, v in self.model.state_dict().items()
            if not k.startswith(
                (
                    "uncertainty_resource_heads",
                    "uncertainty_timing_head",
                    "uncertainty_resource_ensemble_heads",
                    "uncertainty_timing_ensemble_heads",
                    "uncertainty_resource_modulators",
                    "uncertainty_timing_modulators",
                    "uncertainty_resource_quantile_heads",
                    "uncertainty_timing_quantile_head",
                )
            )
        }
        torch.save(backbone_sd, backbone_path)
        if (
            self.model.uncertainty_resource_heads is not None
            or self.model.uncertainty_resource_ensemble_heads is not None
            or self.model.uncertainty_resource_quantile_heads is not None
        ):
            unc_path = os.path.join(save_dir, f"{self.model.name}.uncertainty.pt")
            if self.model.uncertainty_resource_ensemble_heads is not None:
                if self.model.uncertainty_resource_modulators is not None:
                    unc_payload = {
                        "kind": "modulated_ensemble",
                        "resource_ensemble_heads": self.model.uncertainty_resource_ensemble_heads.state_dict(),
                        "timing_ensemble_heads": self.model.uncertainty_timing_ensemble_heads.state_dict(),
                        "resource_modulators": self.model.uncertainty_resource_modulators.state_dict(),
                        "timing_modulators": self.model.uncertainty_timing_modulators.state_dict(),
                        "ensemble_size": self.model.uncertainty_ensemble_size,
                        "calibration": self.uncertainty_calibration,
                        "spread_weight": self.uncertainty_spread_weight,
                    }
                else:
                    unc_payload = {
                        "kind": "ensemble",
                        "resource_ensemble_heads": self.model.uncertainty_resource_ensemble_heads.state_dict(),
                        "timing_ensemble_heads": self.model.uncertainty_timing_ensemble_heads.state_dict(),
                        "ensemble_size": self.model.uncertainty_ensemble_size,
                        "mask_rate": self.model.uncertainty_mask_rate,
                        "calibration": self.uncertainty_calibration,
                        "spread_weight": self.uncertainty_spread_weight,
                    }
            elif self.model.uncertainty_resource_quantile_heads is not None:
                unc_payload = {
                    "kind": "quantile",
                    "resource_quantile_heads": self.model.uncertainty_resource_quantile_heads.state_dict(),
                    "timing_quantile_head": self.model.uncertainty_timing_quantile_head.state_dict(),
                    "calibration": self.uncertainty_calibration,
                    "spread_weight": self.uncertainty_spread_weight,
                }
            else:
                unc_payload = {
                    "kind": "single",
                    "resource_heads": self.model.uncertainty_resource_heads.state_dict(),
                    "timing_head": self.model.uncertainty_timing_head.state_dict(),
                    "calibration": self.uncertainty_calibration,
                    "spread_weight": self.uncertainty_spread_weight,
                }
            torch.save(
                unc_payload,
                unc_path,
            )

    def _load_weights(self, weights_path):
        if not weights_path.endswith(".pt"):
            raise ValueError("Weights filename must end with .pt")
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device), strict=False
        )
        unc_path = weights_path.replace(".weights.pt", ".uncertainty.pt")
        if os.path.isfile(unc_path):
            unc_sd = torch.load(unc_path, map_location=self.device)
            if unc_sd.get("kind") == "modulated_ensemble":
                self.model.add_uncertainty_modulated_ensemble_heads(
                    ensemble_size=unc_sd["ensemble_size"]
                )
                self.model.uncertainty_resource_ensemble_heads.load_state_dict(
                    unc_sd["resource_ensemble_heads"]
                )
                self.model.uncertainty_timing_ensemble_heads.load_state_dict(
                    unc_sd["timing_ensemble_heads"]
                )
                self.model.uncertainty_resource_modulators.load_state_dict(
                    unc_sd["resource_modulators"]
                )
                self.model.uncertainty_timing_modulators.load_state_dict(
                    unc_sd["timing_modulators"]
                )
            elif unc_sd.get("kind") == "ensemble":
                self.model.add_uncertainty_ensemble_heads(
                    ensemble_size=unc_sd["ensemble_size"],
                    mask_rate=unc_sd.get("mask_rate", 0.0),
                )
                self.model.uncertainty_resource_ensemble_heads.load_state_dict(
                    unc_sd["resource_ensemble_heads"]
                )
                self.model.uncertainty_timing_ensemble_heads.load_state_dict(
                    unc_sd["timing_ensemble_heads"]
                )
            elif unc_sd.get("kind") == "quantile":
                self.model.add_uncertainty_quantile_heads()
                self.model.uncertainty_resource_quantile_heads.load_state_dict(
                    unc_sd["resource_quantile_heads"]
                )
                self.model.uncertainty_timing_quantile_head.load_state_dict(
                    unc_sd["timing_quantile_head"]
                )
            else:
                self.model.add_uncertainty_heads()
                self.model.uncertainty_resource_heads.load_state_dict(
                    unc_sd["resource_heads"]
                )
                self.model.uncertainty_timing_head.load_state_dict(unc_sd["timing_head"])
            self.uncertainty_calibration = unc_sd.get("calibration")
            self.uncertainty_spread_weight = unc_sd.get("spread_weight")

    def predict_uncertainty_from_df(self, inputs_df):
        has_single = self.model.uncertainty_resource_heads is not None
        has_ensemble = self.model.uncertainty_resource_ensemble_heads is not None
        has_quantile = self.model.uncertainty_resource_quantile_heads is not None
        if not has_single and not has_ensemble and not has_quantile:
            raise RuntimeError("Uncertainty heads are not initialized.")
        point_estimates = super().predict_from_df(inputs_df.copy())

        feature_labels = list(self.global_numerical_labels) + list(
            self.global_categorical_maps.keys()
        )
        inputs_copy = inputs_df.copy()
        if len(self.sequential_numerical_labels) > 0:
            sequential_labels = list(self.sequential_numerical_labels) + list(
                self.sequential_categorical_maps.keys()
            )
            inputs_copy.loc[:, "sequential_inputs"] = inputs_copy["sequential_inputs"].apply(
                lambda x: x[sequential_labels]
            )
            feature_labels += ["sequential_inputs"]
        inputs_filtered = inputs_copy[feature_labels]
        input_dict = self.build_inputs(inputs_filtered)
        input_tensors = self._process_input_dict(input_dict)
        tensor_dict = {
            k: v.to(self.device)
            for k, v in zip(input_dict.keys(), input_tensors)
        }

        self.model.eval()
        with torch.no_grad():
            if has_quantile:
                _, unc_tensor = self.model.forward_uncertainty_quantiles(tensor_dict)
                raw_uncertainty = unc_tensor.cpu().numpy()
            elif has_ensemble and hasattr(self.model, "forward_uncertainty_stats"):
                mean_unc, std_unc = self.model.forward_uncertainty_stats(tensor_dict)
                raw_uncertainty = self._apply_spread_fusion(
                    mean_unc.cpu().numpy(),
                    std_unc.cpu().numpy(),
                    self.uncertainty_spread_weight,
                )
            else:
                unc_tensor = self.model.forward_uncertainty(tensor_dict)
                raw_uncertainty = unc_tensor.cpu().numpy()
        calibrated_uncertainty = self._apply_monotonic_calibration(
            raw_uncertainty, self.uncertainty_calibration
        )
        return point_estimates, calibrated_uncertainty

    def predict_uncertainty_interval_from_df(self, inputs_df):
        if self.model.uncertainty_resource_quantile_heads is None:
            raise RuntimeError("Quantile uncertainty heads are not initialized.")
        point_estimates = super().predict_from_df(inputs_df.copy())

        feature_labels = list(self.global_numerical_labels) + list(
            self.global_categorical_maps.keys()
        )
        inputs_copy = inputs_df.copy()
        if len(self.sequential_numerical_labels) > 0:
            sequential_labels = list(self.sequential_numerical_labels) + list(
                self.sequential_categorical_maps.keys()
            )
            inputs_copy.loc[:, "sequential_inputs"] = inputs_copy["sequential_inputs"].apply(
                lambda x: x[sequential_labels]
            )
            feature_labels += ["sequential_inputs"]
        inputs_filtered = inputs_copy[feature_labels]
        input_dict = self.build_inputs(inputs_filtered)
        input_tensors = self._process_input_dict(input_dict)
        tensor_dict = {
            k: v.to(self.device)
            for k, v in zip(input_dict.keys(), input_tensors)
        }

        self.model.eval()
        with torch.no_grad():
            lower, upper = self.model.forward_uncertainty_quantiles(tensor_dict)
        return point_estimates, lower.cpu().numpy(), upper.cpu().numpy()


class RoundingWrapper(TorchModelWrapper):
    """
    TorchModelWrapper subclass that rounds resource target predictions
    (BRAM, DSP, FF, LUT — first 4 outputs) to the nearest non-negative integer
    at inference time.

    Motivation: resource targets are integer-valued. The Softplus activation in
    the model head prevents exact-zero predictions. However, many conv1d/conv2d
    networks have zero BRAM or DSP usage (~57%/72% in conv1d, ~71%/81% in conv2d
    test samples). The model's small-positive Softplus outputs for these cases get
    SMAPE=200% at evaluation, regardless of how small the prediction is (as long as
    it is above the eps=1e-10 threshold in the SMAPE formula).

    Rounding converts small Softplus outputs (< 0.5) to 0, eliminating 200% SMAPE
    penalties for zero-valued resource targets whenever the model correctly identifies
    a low-resource configuration. The rounding also helps non-zero integer targets
    (e.g. BRAM=28 predicted as 28.3 → 28).
    """

    # Resource targets: [bram, dsp, ff, lut] = first 4 outputs in ALL_TARGETS order
    _N_RESOURCE = 4

    def predict_from_df(self, inputs_df, **kwargs):
        preds = super().predict_from_df(inputs_df, **kwargs)
        # Round resource predictions to nearest integer, clamp to non-negative
        preds[:, : self._N_RESOURCE] = np.maximum(
            np.round(preds[:, : self._N_RESOURCE]), 0.0
        )
        return preds
