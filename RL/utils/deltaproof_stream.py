import hashlib
import json
import os
from pathlib import Path


class DeltaProofJournal:
    """Persist complete games before updating metrics or accepting more output."""

    def __init__(self, round_dir, requests, progress):
        self.path = Path(round_dir) / 'deltaproof_progress.jsonl'
        self.requests = {request['request_id']: request for request in requests}
        if len(self.requests) != len(requests):
            raise ValueError('DeltaProof request IDs must be unique.')
        self.progress = progress
        self.results = {}
        self.fingerprints = {}
        self.inference = {
            'batch_sizes': [], 'batch_count': 0, 'request_count': 0,
            'queue_wait_seconds': 0.0, 'model_seconds': 0.0,
        }
        # Only an unterminated final record can be discarded after interruption.
        with self.path.open('a+b') as journal:
            journal.seek(0)
            while True:
                offset = journal.tell()
                line = journal.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    journal.truncate(offset)
                    journal.flush()
                    os.fsync(journal.fileno())
                    break
                event = json.loads(line)
                self.validate(event)
                if event['request_id'] in self.results:
                    self.check_duplicate(event)
                    continue
                if event['completion_index'] != len(self.results):
                    raise ValueError('Invalid journal completion order.')
                self.remember(event, log=False)

    def validate(self, event):
        request_id = event['request_id']
        request = self.requests[request_id]
        if event['type'] != 'result' or event['request'] != request:
            raise ValueError('DeltaProof result does not match its request.')
        if any(event[key] != request[key] for key in ('theorem_id', 'source', 'attempt')):
            raise ValueError('DeltaProof result metadata does not match its request.')
        if event['status'] not in ('proved', 'failed', 'rejected'):
            raise ValueError('Invalid DeltaProof result status.')
        if event['status'] == 'proved' and not isinstance(event['proof'], str):
            raise ValueError('Proved DeltaProof result must contain a proof.')
        for key in ('proof', 'error', 'episode_reward', 'duration_seconds', 'timings',
                    'simulations_allocated', 'simulations_used', 'completion_index'):
            event[key]
        timings = event['timings']
        for key in ('total_seconds', 'setup_seconds', 'final_verification'):
            timings[key]
        for key in ('tactic_generation', 'tactic_execution', 'internal_actions'):
            timings[key]['total_seconds']
        if timings['final_verification'] is not None:
            timings['final_verification']['seconds']
        if event['transition_count'] != len(event['transitions']):
            raise ValueError('DeltaProof transition count does not match its result.')
        for index, transition in enumerate(event['transitions']):
            expected = {
                'transition_id': f'{request_id}:{index}',
                'request_id': request_id,
                'batch_id': self.path.parent.name,
                'index': index,
                **{key: request[key] for key in ('theorem_id', 'source', 'attempt')},
            }
            if any(transition[key] != value for key, value in expected.items()):
                raise ValueError('DeltaProof transition metadata does not match its request.')
            for key in ('state', 'action', 'value'):
                transition[key]
        for key in self.inference:
            event['inference'][key]

    def check_duplicate(self, event):
        if self.fingerprint(event) != self.fingerprints[event['request_id']]:
            raise ValueError('Conflicting duplicate DeltaProof result.')

    def fingerprint(self, event):
        record = {key: value for key, value in event.items() if key != 'completion_index'}
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).digest()

    def result_record(self, event):
        return {
            key: value for key, value in event.items()
            if key not in ('type', 'request', 'transitions', 'inference')
        }

    def remember(self, event, log):
        result = self.result_record(event)
        self.results[result['request_id']] = result
        self.fingerprints[result['request_id']] = self.fingerprint(event)
        for key in self.inference:
            self.inference[key] += event['inference'][key]
        if log:
            self.progress.log_alphaproof([result], None)
        else:
            self.progress._update_actor_state(result)
        self.progress.record_result(result, log=log)
        if log:
            self.progress.log(force=True)

    def record_event(self, event):
        if event['type'] == 'summary':
            if len(self.results) != len(self.requests):
                raise ValueError('DeltaProof summary arrived before all results.')
            return
        self.validate(event)
        if event['request_id'] in self.results:
            self.check_duplicate(event)
            return
        event['completion_index'] = len(self.results)
        with self.path.open('a', encoding='utf-8') as journal:
            journal.write(json.dumps(event) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        self.remember(event, log=True)

    def finalize(self):
        if self.results.keys() != self.requests.keys():
            raise ValueError('DeltaProof did not return one result per request.')
        results_path = self.path.parent / 'deltaproof_results.jsonl'
        transitions_path = self.path.parent / 'deltaproof_transitions.jsonl'
        results_tmp = results_path.with_suffix('.jsonl.tmp')
        transitions_tmp = transitions_path.with_suffix('.jsonl.tmp')
        seen = set()
        with self.path.open(encoding='utf-8') as journal, \
                results_tmp.open('w', encoding='utf-8') as results, \
                transitions_tmp.open('w', encoding='utf-8') as transitions:
            for line in journal:
                event = json.loads(line)
                request_id = event['request_id']
                if request_id in seen:
                    continue
                seen.add(request_id)
                results.write(json.dumps(self.results[request_id]) + '\n')
                for transition in event['transitions']:
                    transitions.write(json.dumps(transition) + '\n')
        results_tmp.replace(results_path)
        transitions_tmp.replace(transitions_path)
        count = self.inference['batch_count']
        inference = self.inference | {'average_batch_size': (
            self.inference['request_count'] / count if count else 0.0
        )}
        path = self.path.parent / 'deltaproof_results_metrics.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(inference, indent=2), encoding='utf-8')
        temporary.replace(path)
