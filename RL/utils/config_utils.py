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
    'model_axis_size': int,
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
        'dataset_size': int,
        'total_rounds': int,
        'temperature': NUMBER,
        'epochs': int,
        'sampler': str,
        'conjecture_multiplier': int,
        'llm': {
            'base_model': str,
            'dataset_config': str,
            'samples_per_statement': {
                'first_round': int,
                'later_rounds': int,
            },
        },
    },
    'training': TRAINING_SCHEMA,
}

DELTAPROOF_RL_SCHEMA = {
    'deltaproof': dict,
    'experiment': {
        'type': str,
        'exp_dir': str,
        'dataset_size': int,
        'total_rounds': int,
        'conjecture_multiplier': int,
        'deltaproof': {
            'conjecturer_model': str,
            'vllm_gpu_memory_utilization': NUMBER,
            'conjecturer_sft_dataset': str,
            'conjecturer_sft_ratio': int,
            'conjecturer_temperature': NUMBER,
            'conjecturer_epochs': int,
            'repo_dir': str,
            'python': str,
            'dataset_path': str,
            'lean_project': str,
            'lake_path': str,
            'imports': [str],
            'attempts_per_round': int,
            'conjecture_attempts': int,
            'conjecture_fraction': NUMBER,
            'learner_steps_per_round': int,
        },
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
            'training.trainer.model_axis_size': trainer['model_axis_size'],
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
        positive['experiment.total_rounds'] = experiment['total_rounds']
        if kind == 'rl':
            positive['experiment.conjecture_multiplier'] = experiment['conjecture_multiplier']
            if 'deltaproof' in experiment:
                deltaproof = experiment['deltaproof']
                if not 0 < deltaproof['vllm_gpu_memory_utilization'] <= 1:
                    raise ValueError('experiment.deltaproof.vllm_gpu_memory_utilization must be in (0, 1]')
                if deltaproof['conjecturer_sft_ratio'] < 0:
                    raise ValueError('experiment.deltaproof.conjecturer_sft_ratio must be nonnegative')
                positive |= {
                    'experiment.deltaproof.conjecturer_epochs': deltaproof['conjecturer_epochs'],
                    'experiment.deltaproof.attempts_per_round': deltaproof['attempts_per_round'],
                    'experiment.deltaproof.conjecture_attempts': deltaproof['conjecture_attempts'],
                    'experiment.deltaproof.learner_steps_per_round': deltaproof['learner_steps_per_round'],
                }
                if deltaproof['conjecture_fraction'] not in (0, 0.5):
                    raise ValueError(
                        'experiment.deltaproof.conjecture_fraction must be 0 or 0.5'
                    )
                divisor = 2 * deltaproof['conjecture_attempts']
                if (
                    deltaproof['conjecture_fraction'] == 0.5
                    and deltaproof['attempts_per_round'] % divisor != 0
                ):
                    raise ValueError(
                        'experiment.deltaproof.attempts_per_round must be divisible '
                        'by twice conjecture_attempts'
                    )
            else:
                positive['experiment.epochs'] = experiment['epochs']
                samples = experiment['llm']['samples_per_statement']
                positive |= {
                    'experiment.llm.samples_per_statement.first_round': samples['first_round'],
                    'experiment.llm.samples_per_statement.later_rounds': samples['later_rounds'],
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
    if kind == 'rl' and 'deltaproof' in config['experiment']:
        if config['experiment']['deltaproof']['conjecturer_temperature'] < 0:
            raise ValueError('experiment.deltaproof.conjecturer_temperature must be nonnegative')
    elif kind != 'sft' and config['experiment']['temperature'] < 0:
        raise ValueError('experiment.temperature must be nonnegative')
    expected_samplers = {
        'rl': 'Sampler_base',
        'expert_iteration': 'Sampler_naive',
        'parallel_sampling': 'Sampler_naive',
    }
    if (
        kind in expected_samplers
        and 'sampler' in config['experiment']
        and config['experiment']['sampler'] != expected_samplers[kind]
    ):
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
    if kind == 'rl':
        experiment = config.get('experiment', {})
        solver_keys = {'llm', 'deltaproof'}.intersection(experiment)
        if len(solver_keys) != 1:
            raise ValueError('experiment must contain exactly one of llm or deltaproof')
        if 'deltaproof' in solver_keys:
            schemas['rl'] = DELTAPROOF_RL_SCHEMA
    _validate(config, schemas[kind], kind)
    if kind == 'rl' and 'deltaproof' in config['experiment']:
        exp_dir = Path(config['experiment']['exp_dir'])
        config['experiment']['exp_dir'] = str((REPO_DIR / exp_dir).resolve())
    _validate_ranges(config, kind)
    if kind == 'rl' and 'llm' in config['experiment']:
        dataset_config = Path(config['experiment']['llm']['dataset_config'])
        if not dataset_config.is_absolute():
            config['experiment']['llm']['dataset_config'] = str(REPO_DIR / dataset_config)
    elif kind != 'sft' and 'dataset_config' in config['experiment']:
        dataset_config = Path(config['experiment']['dataset_config'])
        if not dataset_config.is_absolute():
            config['experiment']['dataset_config'] = str(REPO_DIR / dataset_config)
    return config
