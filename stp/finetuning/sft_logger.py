"""Weights & Biases logging for whole-proof supervised fine-tuning."""

from typing import Any, Literal

import wandb


class SFTLogger:
    """Log step-based causal SFT metrics and manage the W&B run lifecycle."""

    def __init__(
        self,
        run_name: str,
        run_id: str,
        wandb_name: str | None,
        wandb_mode: Literal["online", "offline", "disabled"],
        resume: bool,
        config: dict[str, Any],
    ) -> None:
        """Start or resume one W&B run from names, mode, and serialized config."""

        self.run: Any = wandb.init(
            project="alpha-stp",
            name=wandb_name or run_name,
            id=run_id,
            mode=wandb_mode,
            resume="allow" if resume else "never",
            config=config,
            settings=wandb.Settings(
                finish_timeout=60.0,
                finish_timeout_raises=True,
            ),
        )
        self.run.define_metric("sft/step")
        self.run.define_metric("train/*", step_metric="sft/step")
        self.run.define_metric("validation/*", step_metric="sft/step")

    def log_training(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        examples_seen: int,
        target_tokens_seen: int,
    ) -> None:
        """Log one causal optimizer update and its cumulative data counts."""

        self.run.log(
            {
                "sft/step": step,
                "train/loss": loss,
                "train/learning_rate": learning_rate,
                "train/examples_seen": examples_seen,
                "train/target_tokens_seen": target_tokens_seen,
            }
        )

    def log_validation(
        self,
        step: int,
        metrics: dict[str, float | int],
        samples: int,
    ) -> None:
        """Log one fixed-subset or full validation pass."""

        self.run.log(
            {
                "sft/step": step,
                **{
                    f"validation/{name}": value
                    for name, value in metrics.items()
                },
                "validation/samples": samples,
            }
        )

    def log_epoch(
        self,
        step: int,
        epoch: int,
        training_metrics: dict[str, float | int],
        validation_metrics: dict[str, float | int],
        validation_samples: int,
    ) -> None:
        """Log epoch aggregates and the complete validation result."""

        self.run.log(
            {
                "sft/step": step,
                "train/epoch": epoch,
                **{
                    f"train/epoch_{name}": value
                    for name, value in training_metrics.items()
                },
                **{
                    f"validation/{name}": value
                    for name, value in validation_metrics.items()
                },
                "validation/samples": validation_samples,
            }
        )

    def finish(self, exit_code: int) -> None:
        """Close the W&B run with the process exit code."""

        self.run.finish(exit_code=exit_code)
