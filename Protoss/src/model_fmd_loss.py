"""Model-FMD: flow matching distillation from the frozen pretrained model.

Only the masked teacher/student flow-velocity MSE is optimized, with weight 1.
Default zero-output LoRA initialization exactly matches the teacher, so this
objective alone has zero loss/gradient there. No perturbation is introduced.
"""

import torch

from .model_s_forward import flow_preservation_forward


LOSS_ID = "pretrained_flow_distillation-v1"


class ModelFMDLoss:
    def forward(self, model, inputs, diagnostics=False):
        with flow_preservation_forward(model.action_head):
            output = model(**inputs)
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise ValueError("Non-finite Model-FMD distillation loss")
        return loss, {"fmd_loss": loss.detach().item()}

    def state_dict(self):
        return {"format": LOSS_ID}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("Model-FMD requires a flow distillation checkpoint; old FM runs cannot resume")
