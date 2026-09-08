import json
import time
import uuid
from collections import Counter
from pathlib import Path

import wandb

from utils.deltaproof_utils import write_jsonl
from utils.RL_utils import CONJECTURE_THRESHOLD


class ConjectureMetrics:
    def __init__(self, config, round_dir, model):
        self.round_dir = Path(round_dir)
        self.round_id = int(self.round_dir.name.removeprefix('round'))
        experiment_dir = self.round_dir.parent
        run_id_path = experiment_dir / 'conjecturer_wandb_id.txt'
        if not run_id_path.exists():
            run_id_path.write_text(uuid.uuid4().hex, encoding='utf-8')
        tracker = config['training']['trainer']['tracker']
        self.run = wandb.init(
            entity=tracker['entity'],
            project=tracker['project'],
            name=tracker['name'] + '-conjecturer-metrics',
            tags=tracker['tags'],
            id=run_id_path.read_text(encoding='utf-8').strip(),
            resume='allow',
            dir=str(experiment_dir),
            config=config,
        )
        self.metadata = {'conjecturer_model': str(model)}
        self.metrics = {**dict.fromkeys([
            'generated', 'distinct_new', 'lean_checked', 'lean_passed',
            'selected', 'attempts_expected', 'attempts_completed',
            'attempts_solved', 'solved_at_least_once', 'fully_evaluated',
            'unsolved', 'hard', 'easy',
        ], 0), **dict.fromkeys([
            'lean_pass_rate', 'attempt_success_rate', 'solve_rate',
            'unsolved_fraction', 'hard_fraction', 'easy_fraction',
            'training_examples',
        ])}
        self.records = []
        self.by_lemma = {}
        self.requests = {}
        self.expected = Counter()
        self.completed = Counter()
        self.solved = Counter()
        self.results = {}
        self.last_log = 0
        self.log(force=True)

    def log(self, force=False):
        now = time.monotonic()
        if force or now - self.last_log >= 30:
            self.run.log({
                'round': self.round_id,
                **self.metadata,
                **{
                    f'conjecturer/{key}': value
                    for key, value in self.metrics.items()
                    if value is not None
                },
            })
            self.last_log = now

    def generation_progress(self, completed, total):
        self.metrics.update(generated=completed, generation_target=total)
        self.log(force=completed == total)

    def set_candidates(self, candidates, distinct):
        retained = {id(candidate) for candidate in distinct}
        for index, candidate in enumerate(candidates):
            record = {
                'candidate_id': index,
                'lemma_id': candidate['lemma_id'],
                'statement': candidate['statement'],
                'header': candidate['header'],
                'distinct_new': id(candidate) in retained,
                'lean_passed': None,
                'selected': False,
                'attempts': 0,
                'solved_attempts': 0,
                'solve_rate': None,
            }
            self.records.append(record)
            if record['distinct_new']:
                self.by_lemma[record['lemma_id']] = record
        self.metrics.update(generated=len(candidates), distinct_new=len(distinct))
        self.log(force=True)

    def validation_progress(self, results):
        for result in results:
            passed = result.get('pass', False)
            self.by_lemma[result['lemma_id']]['lean_passed'] = passed
            self.metrics['lean_checked'] += 1
            self.metrics['lean_passed'] += int(passed)
        if self.metrics['lean_checked']:
            self.metrics['lean_pass_rate'] = (
                self.metrics['lean_passed'] / self.metrics['lean_checked']
            )
        self.log()

    def select(self, conjectures):
        for conjecture in conjectures:
            self.by_lemma[conjecture['lemma_id']]['selected'] = True
        self.metrics['selected'] = len(conjectures)
        self.log(force=True)

    def set_requests(self, requests, solver_run_dir):
        self.requests = {item['request_id']: item for item in requests}
        self.expected = Counter(
            item['theorem_id'] for item in requests if item['source'] == 'conjecture'
        )
        self.metrics['attempts_expected'] = sum(self.expected.values())
        self.metadata['solver_run_dir'] = str(solver_run_dir)
        self.log(force=True)

    def record_result(self, result):
        request_id = result['request_id']
        status = result['status']
        if request_id in self.results:
            assert self.results[request_id] == status
            return
        self.results[request_id] = status
        request = self.requests[request_id]
        if request['source'] != 'conjecture':
            return
        theorem_id = request['theorem_id']
        self.completed[theorem_id] += 1
        self.solved[theorem_id] += int(status == 'proved')
        finished = [
            key for key in self.expected if self.completed[key] == self.expected[key]
        ]
        rates = [self.solved[key] / self.expected[key] for key in finished]
        completed = sum(self.completed.values())
        solved = sum(self.solved.values())
        self.metrics.update(
            attempts_completed=completed,
            attempts_solved=solved,
            attempt_success_rate=solved / completed,
            solved_at_least_once=sum(value > 0 for value in self.solved.values()),
            fully_evaluated=len(finished),
            solve_rate=sum(rate > 0 for rate in rates) / len(rates) if rates else None,
            unsolved=sum(rate == 0 for rate in rates),
            hard=sum(0 < rate <= CONJECTURE_THRESHOLD for rate in rates),
            easy=sum(rate > CONJECTURE_THRESHOLD for rate in rates),
        )
        for bucket in ('unsolved', 'hard', 'easy'):
            self.metrics[bucket + '_fraction'] = (
                self.metrics[bucket] / len(finished) if finished else None
            )
        self.log()

    def save(self, round_metrics):
        for lemma_id, record in self.by_lemma.items():
            theorem_id = f'conjecture:{lemma_id}'
            attempts = self.completed[theorem_id]
            solved = self.solved[theorem_id]
            record.update(
                attempts=attempts,
                expected_attempts=self.expected[theorem_id],
                fully_evaluated=record['selected'] and attempts == self.expected[theorem_id],
                solved_attempts=solved,
                solve_rate=solved / attempts if attempts else None,
            )
        write_jsonl(self.round_dir / 'conjecture_metrics.jsonl', self.records)
        metrics = round_metrics | self.metadata | {
            f'conjecturer/{key}': value for key, value in self.metrics.items()
        }
        (self.round_dir / 'round_metrics.json').write_text(
            json.dumps(metrics, indent=2), encoding='utf-8',
        )
        self.log(force=True)
