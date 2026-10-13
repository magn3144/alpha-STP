import hashlib
import json
import time
from pathlib import Path

from utils.file_utils import read_file
from utils.RL_utils import get_deduplication_key
from utils.training_data_utils import get_conjecture_level
from utils.stp_wandb import initialize_run


class LLMMetrics:
    """Track verified solve attempts, including repeated identical proofs."""

    def __init__(self, config, round_dir):
        self.round_dir = Path(round_dir)
        self.round_id = int(self.round_dir.name.removeprefix('round'))
        self.path = self.round_dir / 'actor_metrics.json'
        self.records = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.previous_games = 0
        self.proved_theorems = set()
        for previous_round in range(self.round_id):
            proofs = read_file(str(self.round_dir.parent / f'round{previous_round}'
                                   / 'sampler_ckpt/generated_proofs.json'))
            self.previous_games += sum(proof.get('multiplicity', 1) for proof in proofs)
            self.proved_theorems.update(
                proof['lemma_id'] for proof in proofs
                if proof['complete'] and get_conjecture_level(proof) == 0
            )
        self.run = initialize_run(config, round_dir)
        self.last_log = 0
        self.update([], {}, force=True)

    def update(self, proofs, results, force=False):
        now = time.monotonic()
        if not force and now - self.last_log < 30:
            return
        for proof in proofs:
            key = get_deduplication_key(proof)
            result = proof if 'complete' in proof else results.get(key, proof)
            if 'complete' not in result:
                continue
            fingerprint = hashlib.sha256(json.dumps(key).encode()).hexdigest()
            self.records[fingerprint] = {
                'games': max(proof.get('multiplicity', 1), self.records.get(fingerprint, {}).get('games', 0)),
                'complete': result['complete'],
                'dataset': get_conjecture_level(proof) == 0,
                'lemma_id': proof['lemma_id'],
            }
        games = self.previous_games + sum(record['games'] for record in self.records.values())
        proved = self.proved_theorems | {
            record['lemma_id'] for record in self.records.values()
            if record['complete'] and record['dataset']
        }
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.records))
        temporary.replace(self.path)
        self.run.log({
            'round': self.round_id,
            'actor/game': games,
            'actor/unique_theorems_proved': len(proved),
        })
        self.last_log = now

    def finish(self):
        self.run.finish()
