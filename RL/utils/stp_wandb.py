import uuid
from pathlib import Path

import wandb


def initialize_run(config, round_dir):
    experiment_dir = Path(round_dir).parent
    filename = 'actor_wandb_id.txt' if 'llm' in config['experiment'] else 'conjecturer_wandb_id.txt'
    run_id_path = experiment_dir / filename
    if not run_id_path.exists():
        run_id_path.write_text(uuid.uuid4().hex)
    tracker = config['training']['trainer']['tracker']
    run = wandb.init(
        entity=tracker['entity'], project=tracker['project'],
        name=tracker['name'], tags=tracker['tags'],
        id=run_id_path.read_text().strip(), resume='allow',
        dir=str(experiment_dir), config=config, allow_val_change=True,
    )
    run.define_metric('actor/game')
    run.define_metric('actor/*', step_metric='actor/game')
    run.define_metric('round')
    run.define_metric('combined/*', step_metric='round')
    run.define_metric('RL_new/*', step_metric='round')
    run.define_metric('monitoring/*', step_metric='round')
    return run
