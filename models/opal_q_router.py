from typing import Dict

from .router import LayerRouter


class OPALQRiskRouter(LayerRouter):
    """Small prefix-hidden-state router for OPAL-Q Lite.

    It keeps the existing LayerRouter training behavior but names the outputs in
    OPAL-Q terms: high score means high risk if skipped, so the mask keeps the
    highest-risk layers at a fixed skip rate.
    """

    def risk_outputs(self, hidden_state) -> Dict[str, object]:
        mask, scores = self(hidden_state)
        return {
            "mask": mask,
            "keep_score": scores,
            "skip_risk": scores,
        }

