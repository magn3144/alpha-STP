"""Run unchanged Levanter training with persistent STP experiment logging."""

import logging
from dataclasses import dataclass

import levanter.config
from levanter.tracker.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig, WandbTracker
from weighted_lm import train


class STPWandbTracker(WandbTracker):
    def __init__(self, run, round_id, metric_prefix):
        super().__init__(run)
        self.round_id = round_id
        self.metric_prefix = metric_prefix
        self.step_metric = metric_prefix + 'learner/step'
        self.offset = int(run.summary.get(self.step_metric, 0))
        self.local_step = -1
        self.defined_metrics = set()
        run.define_metric(self.step_metric)

    def log(self, metrics, *, step, commit=None):
        metrics = {self.metric_prefix + key: value for key, value in metrics.items()}
        for key in metrics.keys() - self.defined_metrics:
            self.run.define_metric(key, step_metric=self.step_metric)
            self.defined_metrics.add(key)
        if step is not None:
            if int(step) > self.local_step and self.local_step >= 0:
                self.run.log({}, commit=True)
            self.local_step = max(self.local_step, int(step))
        metrics |= {'round': self.round_id, self.step_metric: self.offset + self.local_step + 1}
        # W&B's history index also includes actor events; never use the local
        # optimizer step as its index, or resumed-round metrics get discarded.
        self.run.log(metrics, commit=False if commit is None else commit)


@TrackerConfig.register_subclass('stp_wandb')
@dataclass
class STPWandbConfig(WandbConfig):
    round_id: int = 0
    metric_prefix: str = ''

    def init(self, run_id):
        tracker = super().init(run_id)
        return STPWandbTracker(tracker.run, self.round_id, self.metric_prefix)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    levanter.config.main(train)()
