import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
RL_DIR = REPO_DIR / 'RL'


def run_python(script, *args, cwd=RL_DIR, env=None, dry_run=False):
    command = [sys.executable, '-u', str(script), *(str(arg) for arg in args)]
    print(f'+ {shlex.join(command)}', flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def levanter_environment():
    env = os.environ.copy()
    levanter_dir = REPO_DIR / 'levanter'
    python_paths = [levanter_dir, levanter_dir / 'src', levanter_dir / 'examples']
    if env.get('PYTHONPATH'):
        python_paths.append(Path(env['PYTHONPATH']))
    env['PYTHONPATH'] = os.pathsep.join(str(path) for path in python_paths)
    return env
