"""Restore Aether's saved topology before loading its inference parameters."""

from traiNNer.models.sr_qat import load_aether_for_inference as load_deployment_weights

__all__ = ["load_deployment_weights"]
