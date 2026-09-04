import os
from pathlib import Path

import yaml


NUMBER = (int, float)
REPO_DIR = Path(__file__).resolve().parents[2]

OPTIMIZER_SCHEMA = {
    'learning_rate': NUMBER,
    'weight_decay': NUMBER,
    'beta1': NUMBER,
    'beta2': NUMBER,
    'epsilon': NUMBER,
    'max_grad_norm': NUMBER,
    'lr_schedule': str,
    'stable': NUMBER,
    'cooldown': NUMBER,
    'min_lr_ratio': NUMBER,
}

TRAINER_SCHEMA = {
    'ray': {
        'auto_start_cluster': bool,
    },
    'tracker': {
        'type': str,
        'entity': str,
        'project': str,
        'name': str,
        'tags': [str],
    },
    'seed': int,
    'mp': str,
    'train_batch_size': int,
    'steps_per_eval': int,
    'tensor_parallel_axes': [str],
    'per_device_eval_parallelism': int,
    'per_device_parallelism': int,
}

TRAINING_SCHEMA = {
    'max_tune_length': int,
    'trust_remote_code': bool,
    'trainer': TRAINER_SCHEMA,
    'eval_on_first_step': bool,
    'optimizer': OPTIMIZER_SCHEMA,
}

SFT_SCHEMA = {
    'experiment': {
        'type': str,
        'storage': str,
        'base_model': str,
    },
    'training': {
        **TRAINING_SCHEMA,
        'trainer': {
            **TRAINER_SCHEMA,
            'num_train_steps': int,
        },
        'save_freq': int,
        'optimizer': {
            **OPTIMIZER_SCHEMA,
            'warmup': int,
        },
    },
}

RL_SCHEMA = {
    'experiment': {
        'type': str,
        'exp_dir': str,
        'base_model': str,
        'dataset_config': str,
        'dataset_size': int,
        'total_rounds': int,
        'samples_per_statement': {
            'first_round': int,
            'later_rounds': int,
        },
        'temperature': NUMBER,
        'epochs': int,
        'sampler': str,
        'conjecture_multiplier': int,
    },
    'training': TRAINING_SCHEMA,
}

EXPERT_ITERATION_SCHEMA = {
    'experiment': {
        'type': str,
        'exp_dir': str,
        'base_model': str,
        'dataset_config': str,
        'dataset_size': int,
        'total_rounds': int,
        'seed': int,
        'samples_per_statement': int,
        'temperature': NUMBER,
        'epochs': int,
        'sampler': str,
    },
    'training': TRAINING_SCHEMA,
}

PARALLEL_SAMPLING_SCHEMA = {
    'experiment': {
        'type': str,
        'exp_dir': str,
        'base_model': str,
        'dataset_config': str,
        'dataset_size': int,
        'total_rounds': int,
        'seed': int,
        'samples_per_statement': int,
        'temperature': NUMBER,
        'sampler': str,
    },
}


def _expand_environment(value):
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if '$' in expanded:
            raise ValueError(f'Environment variable in {value!r} is not set')
        return expanded
    return value


def _validate(value, schema, path):
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ValueError(f'{path} must be a mapping')
        missing = schema.keys() - value.keys()
        unknown = value.keys() - schema.keys()
        if missing:
            raise ValueError(f'{path} is missing: {", ".join(sorted(missing))}')
        if unknown:
            raise ValueError(f'{path} has unknown values: {", ".join(sorted(unknown))}')
        for key, item_schema in schema.items():
            _validate(value[key], item_schema, f'{path}.{key}')
        return

    if isinstance(schema, list):
        if not isinstance(value, list) or not all(isinstance(item, schema[0]) for item in value):
            raise ValueError(f'{path} must be a list of {schema[0].__name__} values')
        return

    if schema == NUMBER:
        if isinstance(value, bool) or not isinstance(value, NUMBER):
            raise ValueError(f'{path} must be a number')
        return

    if type(value) is not schema:
        raise ValueError(f'{path} must be {schema.__name__}')


def _validate_ranges(config, kind):
    positive = {}
    if 'training' in config:
        training = config['training']
        trainer = training['trainer']
        optimizer = training['optimizer']
        positive |= {
            'training.max_tune_length': training['max_tune_length'],
            'training.trainer.train_batch_size': trainer['train_batch_size'],
            'training.trainer.steps_per_eval': trainer['steps_per_eval'],
            'training.trainer.per_device_eval_parallelism': trainer['per_device_eval_parallelism'],
            'training.trainer.per_device_parallelism': trainer['per_device_parallelism'],
            'training.optimizer.learning_rate': optimizer['learning_rate'],
            'training.optimizer.epsilon': optimizer['epsilon'],
        }
        if kind == 'sft':
            positive |= {
                'training.trainer.num_train_steps': trainer['num_train_steps'],
                'training.save_freq': training['save_freq'],
            }

    if kind != 'sft':
        experiment = config['experiment']
        positive |= {
            'experiment.total_rounds': experiment['total_rounds'],
        }
        if kind == 'rl':
            positive |= {
                'experiment.samples_per_statement.first_round': experiment['samples_per_statement']['first_round'],
                'experiment.samples_per_statement.later_rounds': experiment['samples_per_statement']['later_rounds'],
                'experiment.epochs': experiment['epochs'],
                'experiment.conjecture_multiplier': experiment['conjecture_multiplier'],
            }
        else:
            positive['experiment.samples_per_statement'] = experiment['samples_per_statement']
            if kind == 'expert_iteration':
                positive['experiment.epochs'] = experiment['epochs']
        if kind in ('expert_iteration', 'parallel_sampling') and experiment['seed'] < 0:
            raise ValueError('experiment.seed must be nonnegative')

    for path, value in positive.items():
        if value <= 0:
            raise ValueError(f'{path} must be greater than zero')

    if 'training' in config:
        if trainer['train_batch_size'] % trainer['per_device_parallelism'] != 0:
            raise ValueError('training.trainer.train_batch_size must be divisible by per_device_parallelism')
        if trainer['seed'] < 0:
            raise ValueError('training.trainer.seed must be nonnegative')
        if not 0 <= optimizer['weight_decay']:
            raise ValueError('training.optimizer.weight_decay must be nonnegative')
        for key in ('beta1', 'beta2'):
            if not 0 <= optimizer[key] < 1:
                raise ValueError(f'training.optimizer.{key} must be in [0, 1)')
        if not 0 < optimizer['min_lr_ratio'] <= 1:
            raise ValueError('training.optimizer.min_lr_ratio must be in (0, 1]')
        if optimizer['max_grad_norm'] <= 0:
            raise ValueError('training.optimizer.max_grad_norm must be greater than zero')
        if optimizer['lr_schedule'] not in ('constant', 'cosine', 'linear', 'inv_sqrt'):
            raise ValueError('training.optimizer.lr_schedule has an unsupported value')
        if optimizer['stable'] < 0 or optimizer['cooldown'] < 0:
            raise ValueError('training.optimizer.stable and cooldown must be nonnegative')
    if kind == 'sft' and optimizer['warmup'] < 0:
        raise ValueError('training.optimizer.warmup must be nonnegative')
    if kind == 'sft' and training['save_freq'] >= trainer['num_train_steps']:
        raise ValueError('training.save_freq must be less than trainer.num_train_steps')
    if kind != 'sft' and config['experiment']['dataset_size'] < 0:
        raise ValueError('experiment.dataset_size must be nonnegative')
    if kind != 'sft' and config['experiment']['temperature'] < 0:
        raise ValueError('experiment.temperature must be nonnegative')
    expected_samplers = {
        'rl': 'Sampler_base',
        'expert_iteration': 'Sampler_naive',
        'parallel_sampling': 'Sampler_naive',
    }
    if kind in expected_samplers and config['experiment']['sampler'] != expected_samplers[kind]:
        raise ValueError(f'experiment.sampler must be {expected_samplers[kind]}')


def load_experiment_config(path, kind=None):
    schemas = {
        'sft': SFT_SCHEMA,
        'rl': RL_SCHEMA,
        'expert_iteration': EXPERT_ITERATION_SCHEMA,
        'parallel_sampling': PARALLEL_SAMPLING_SCHEMA,
    }
    with open(path) as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError('Experiment config must be a mapping')
    config = _expand_environment(config)
    config_kind = config.get('experiment', {}).get('type')
    if kind is not None and config_kind != kind:
        raise ValueError(f'Expected a {kind} config, got {config_kind!r}')
    kind = config_kind
    if kind not in schemas:
        raise ValueError(f'Unknown experiment kind: {kind}')
    _validate(config, schemas[kind], kind)
    _validate_ranges(config, kind)
    if kind != 'sft':
        dataset_config = Path(config['experiment']['dataset_config'])
        if not dataset_config.is_absolute():
            config['experiment']['dataset_config'] = str(REPO_DIR / dataset_config)
    return config
