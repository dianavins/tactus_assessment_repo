"""
Exponential Moving Average (EMA) of model weights.

Maintains a shadow copy of parameters updated as:
    shadow = decay * shadow + (1 - decay) * live_param

The EMA model is used for evaluation and as the base for QAT.
It is never used for the backward pass.

Usage:
    ema = EMAModel(model, decay=0.9998)

    # After each optimiser step:
    ema.update(model)

    # For evaluation:
    with ema.apply_shadow(model):
        val_acc = evaluate(model, val_loader)
    # model weights automatically restored after the with-block
"""

from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.nn as nn


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        """
        Args:
            model: the live model whose parameters will be tracked
            decay: EMA decay factor ceiling (higher = slower update, smoother average)
                   0.9998 is appropriate for large-scale training; the actual decay
                   ramps up from ~0 via warmup so early steps track the model closely.
        """
        self.decay = decay
        self.num_updates = 0
        # Deep-copy the initial parameters as the shadow
        self.shadow: dict[str, torch.Tensor] = {
            name: param.data.clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: nn.Module):
        """Apply one EMA step with warmup: shadow = d * shadow + (1-d) * live.

        Uses the warmup formula:
            effective_decay = min(decay, (1 + num_updates) / (10 + num_updates))

        This ramps from ~0.09 toward `decay` as training progresses.
        Without warmup, high decay values (e.g. 0.9998) cause EMA to stay near
        the random initialisation for hundreds of steps — catastrophic on small
        datasets like DTD where Stage 1 has only ~885 total update steps.
        """
        self.num_updates += 1
        effective_decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
        )
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone().float()
                else:
                    self.shadow[name] = (
                        effective_decay * self.shadow[name]
                        + (1.0 - effective_decay) * param.data.float()
                    )

    @contextmanager
    def apply_shadow(self, model: nn.Module):
        """
        Context manager: temporarily replace model weights with EMA shadow.
        Restores original weights on exit.

        Example:
            with ema.apply_shadow(model):
                acc = evaluate(model, loader)
        """
        # Save original weights
        original: dict[str, torch.Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Apply shadow
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.data.copy_(self.shadow[name].to(param.dtype))

        try:
            yield model
        finally:
            # Restore original weights
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in original:
                        param.data.copy_(original[name])

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay, "num_updates": self.num_updates}

    def load_state_dict(self, state: dict):
        self.shadow      = state["shadow"]
        self.decay       = state.get("decay", self.decay)
        self.num_updates = state.get("num_updates", 0)


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = nn.Linear(8, 4)
    ema   = EMAModel(model, decay=0.9)

    # Record initial EMA shadow
    initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

    # Simulate a gradient step that changes model weights
    with torch.no_grad():
        for p in model.parameters():
            p.data.fill_(1.0)

    ema.update(model)

    # Shadow should have moved toward 1.0 but not reached it
    for name, shadow_val in ema.shadow.items():
        assert not torch.allclose(shadow_val, initial_shadow[name]), \
            "EMA shadow did not update"
        assert not torch.all(shadow_val == 1.0), \
            "EMA shadow should not equal live weights immediately"
    print("EMA update ✓")

    # apply_shadow should not alter weights after context exits
    original_weights = {
        name: param.data.clone()
        for name, param in model.named_parameters()
    }
    with ema.apply_shadow(model):
        # Inside: weights are EMA shadow
        pass
    # Outside: weights must be restored
    for name, param in model.named_parameters():
        assert torch.allclose(param.data, original_weights[name]), \
            f"Weight '{name}' was not restored after apply_shadow"
    print("apply_shadow restore ✓")

    # State dict round-trip
    sd = ema.state_dict()
    ema2 = EMAModel(model, decay=0.5)
    ema2.load_state_dict(sd)
    assert ema2.decay == ema.decay
    print("EMA state_dict round-trip ✓")

    print("All EMA smoke tests passed.")
