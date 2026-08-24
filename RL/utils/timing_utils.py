import argparse
import atexit
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


TIMING_FILE_ENV = 'STP_TIMING_FILE'
TIMING_SESSION_ENV = 'STP_TIMING_SESSION'
TIMING_ROUND_ENV = 'STP_TIMING_ROUND'
_write_lock = threading.Lock()
_summary_registered = False


def configure_timing(exp_dir, new_session=False, round_id=None):
    global _summary_registered

    if new_session or TIMING_SESSION_ENV not in os.environ:
        os.environ[TIMING_SESSION_ENV] = uuid.uuid4().hex
    if new_session or TIMING_FILE_ENV not in os.environ:
        os.environ[TIMING_FILE_ENV] = str(Path(exp_dir) / 'timings.jsonl')
    if round_id is not None:
        os.environ[TIMING_ROUND_ENV] = str(round_id)

    Path(os.environ[TIMING_FILE_ENV]).parent.mkdir(parents=True, exist_ok=True)
    if not _summary_registered:
        atexit.register(write_summary)
        _summary_registered = True


def record_event(event, duration_seconds, status='success', **metadata):
    record = {
        'session': os.environ[TIMING_SESSION_ENV],
        'event': event,
        'duration_seconds': duration_seconds,
        'status': status,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    if TIMING_ROUND_ENV in os.environ:
        record['round'] = int(os.environ[TIMING_ROUND_ENV])
    record.update({key: value for key, value in metadata.items() if value is not None})

    line = json.dumps(record, separators=(',', ':')) + '\n'
    with _write_lock:
        with open(os.environ[TIMING_FILE_ENV], 'a') as timing_file:
            timing_file.write(line)
            timing_file.flush()


class EventTimer:
    def __init__(self, event, **metadata):
        self.event = event
        self.metadata = metadata
        self.start_time = time.perf_counter()
        self.stopped = False

    def stop(self, status='success', **metadata):
        if self.stopped:
            return
        self.stopped = True
        self.metadata.update(metadata)
        record_event(self.event, time.perf_counter() - self.start_time, status, **self.metadata)


@contextmanager
def timer(event, **metadata):
    event_timer = EventTimer(event, **metadata)
    try:
        yield event_timer
    except BaseException as error:
        event_timer.stop('failed', error=type(error).__name__)
        raise
    else:
        event_timer.stop()


def percentile(values, fraction):
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def build_summary(timing_file):
    groups = defaultdict(list)
    with open(timing_file) as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (
                record['session'],
                record['event'],
                record.get('round'),
                record.get('phase'),
                record['status'],
            )
            groups[key].append(record['duration_seconds'])

    summary = []
    for (session, event, round_id, phase, status), durations in sorted(
        groups.items(), key=lambda item: tuple('' if value is None else str(value) for value in item[0])
    ):
        durations.sort()
        result = {
            'session': session,
            'event': event,
            'status': status,
            'count': len(durations),
            'total_seconds': sum(durations),
            'mean_seconds': sum(durations) / len(durations),
            'min_seconds': durations[0],
            'p50_seconds': percentile(durations, 0.50),
            'p95_seconds': percentile(durations, 0.95),
            'max_seconds': durations[-1],
        }
        if round_id is not None:
            result['round'] = round_id
        if phase is not None:
            result['phase'] = phase
        summary.append(result)
    return {'groups': summary}


def write_summary(timing_file=None, summary_file=None):
    timing_file = Path(timing_file or os.environ[TIMING_FILE_ENV])
    if not timing_file.exists():
        return
    summary_file = Path(summary_file or timing_file.with_name('timing_summary.json'))
    summary = build_summary(timing_file)
    temporary_file = summary_file.with_suffix(summary_file.suffix + '.tmp')
    with _write_lock:
        with open(temporary_file, 'w') as output:
            json.dump(summary, output, indent=2)
            output.write('\n')
        os.replace(temporary_file, summary_file)


def main():
    parser = argparse.ArgumentParser(description='Rebuild an STP timing summary.')
    parser.add_argument('path', help='Experiment directory or timings.jsonl path')
    args = parser.parse_args()
    path = Path(args.path)
    timing_file = path / 'timings.jsonl' if path.is_dir() else path
    write_summary(timing_file)


if __name__ == '__main__':
    main()
