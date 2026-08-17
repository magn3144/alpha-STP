import os
import time
import json
import psutil
import tempfile
import subprocess
from typing import List
import ray
from ray._private.utils import hex_to_binary
from ray._raylet import PlacementGroupID
from ray.util import placement_group, remove_placement_group
from ray.util.placement_group import PlacementGroup
from utils.prover.lean.ast_parser import lean4_parser
from func_timeout import FunctionTimedOut, func_set_timeout

__DEBUG__ = os.getenv("DEBUG", 'False').lower() in ('true', '1', 't')
HOME_DIR = os.path.expanduser('~')
DEFAULT_LAKE_PATH = f'{HOME_DIR}/.elan/bin/lake'
DEFAULT_LEAN_WORKSPACE = f'{HOME_DIR}/lean/mathlib4/'
DEFAULT_TIMEOUT = 200
LEAN_HEADER = 'import miniF2F\nimport Aesop\nset_option maxHeartbeats 0\nopen BigOperators Real Nat Topology Rat\n'
TEST_BATCH_SIZE = 40

MEMORY_THRESHOLD = 75.0  # Memory usage percentage to trigger waiting

def extract_invokes(ast_results):
    premises = ast_results.get('premises', [])
    invokes = set()
    for premise in premises:
        invokes.add(premise['fullName'])
    return list(invokes)

def get_result_from_repl(repl_result, code, start_time):
    result = {
        "sorries" : repl_result.get('sorries', []), 
        "tactics" : repl_result.get('tactics', []),
        "errors" : [m for m in repl_result.get('messages', []) if m['severity'] == 'error'],
        "warnings" : [m for m in repl_result.get('messages', []) if m['severity'] == 'warning'],
        "infos" : [m for m in repl_result.get('messages', []) if m['severity'] == 'info'],
        "verified_code" : code,
    }
    result['pass'] = not result['errors']
    result['complete'] = result['pass'] and not result['sorries'] and not any("declaration uses 'sorry'" in warning['data'] or 'failed' in warning['data'] for warning in result['warnings'])
    if result['complete']:
        ast_results = lean4_parser(code, repl_result['ast']) if 'ast' in repl_result and repl_result['ast'] else {}
        result['invokes'] = extract_invokes(ast_results)
        if __DEBUG__:
            result['ast'] = ast_results
    result['verify_time'] = time.time() - start_time
    return result

def read_from_repl(proc):
    ret = ''
    while True:
        line = proc.stdout.readline()
        if len(line.strip()) == 0:
            break
        ret += line
    return ret

@func_set_timeout(DEFAULT_TIMEOUT, allowOverride=True)
def query_repl(proc, message_str):
    proc.stdin.write(message_str)
    proc.stdin.flush()
    return read_from_repl(proc)

@func_set_timeout(DEFAULT_TIMEOUT + 10, allowOverride=True)
def _start_repl_process(lake_path, lean_workspace, header = None):
    proc = subprocess.Popen([lake_path, "exe", 'repl'], 
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,  # Capture stderr
                                    text=True, 
                                    cwd=lean_workspace,)
    cmd = json.dumps({"cmd": header or LEAN_HEADER, "allTactics": False, "ast": False, "tactics": False, "premises": False}, ensure_ascii=False) + '\r\n\r\n'
    query_repl(proc, cmd)
    return proc

def start_repl_process(lake_path, lean_workspace, header = None):
    # Retry if the process is not started
    for i in range(5):
        try:
            return _start_repl_process(lake_path, lean_workspace, header)
        except Exception as e:
            if __DEBUG__:
                print(f"Error in starting Lean4 process: {e}")
            time.sleep(i + 1)
            continue
    raise Exception("Failed to start Lean4 process")

@func_set_timeout(DEFAULT_TIMEOUT, allowOverride=True)
def terminate_repl(proc):
    if proc is None:
        return
    
    try:
        # Create a psutil Process instance for the main process
        parent = psutil.Process(proc.pid)
        
        # Retrieve all child processes recursively
        children = parent.children(recursive=True)
        
        # Terminate all child processes
        for child in children:
            child.terminate()
        
        # Terminate the main process
        parent.terminate()
        
        # Wait for all processes to terminate gracefully
        gone, alive = psutil.wait_procs([parent] + children, timeout=5)
        
        # Force kill any processes that are still alive after the timeout
        for p in alive:
            p.kill()
            
    except psutil.NoSuchProcess:
        # The process may have already terminated
        pass
    except Exception as e:
        # Optionally log the exception if needed
        # print(f"Error in terminating processes: {e}")
        pass

def verify_lean4_file(codes, headers, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, last_env=None, verbose=False, 
                      allTactics=False, ast=False, premises=False, tactics=False):
    command = dict(allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
    
    results = []
    try:
        proc = None
        last_header = None
        for code, header in zip(codes, headers):
            if proc is None or header != last_header:
                terminate_repl(proc)
                proc = start_repl_process(lake_path, lean_workspace, header)
                last_header = header
            
            message_str = json.dumps(command | {'cmd': code, 'env': 0}, ensure_ascii=False) + '\r\n\r\n'
            try:
                start_time = time.time()
                output = query_repl(proc, message_str)
                repl_result = json.loads(output)
                result = get_result_from_repl(repl_result, code, start_time)
                results.append(result)
            except (Exception, FunctionTimedOut) as e:
                if __DEBUG__:
                    print(e)
                results.append({"system_messages": str(e), 'complete': False})
                terminate_repl(proc)
                proc = None

        terminate_repl(proc)
    except (Exception, FunctionTimedOut) as e:
        if __DEBUG__:
            print(e)
        results += [{"system_messages": str(e)}] * (len(codes) - len(results))

    assert len(results) == len(codes), f"Results length mismatch: {len(results)} != {len(codes)}"
    return results

def verify_lean4_file_premises(code, header, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, last_env=None, verbose=False, 
                      timeout=DEFAULT_TIMEOUT, allTactics=False, ast=False, premises=False, tactics=False):
    command = dict(allTactics=allTactics, ast=ast, tactics=tactics, premises=premises)
    if last_env is not None:
        command.update(env=last_env)

    message_str = json.dumps(command | {'cmd': (header or LEAN_HEADER) + code}, ensure_ascii=False) + '\r\n\r\n'
    if verbose:
        print(message_str)
    start_time = time.time()
    
    results = []
    try:
        with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            outputs = subprocess.run([lake_path, "exe", 'repl'], 
                                     stdin=temp_file, 
                                     capture_output=True, 
                                     text=True, 
                                     cwd=lean_workspace, 
                                     timeout=timeout,)

        repl_result = json.loads(outputs.stdout)
        result = get_result_from_repl(repl_result, code, start_time)
        results.append(result)
        return results
    except Exception as e:
        if __DEBUG__:
            print(e)
        return [{"system_messages": str(e), 'complete': False}]

@ray.remote(num_cpus=1)
class Lean4Worker():
    def __init__(self, node, idx, collect_premises=True):
        super().__init__()
        self.node = node
        self.idx = idx
        self.collect_premises = collect_premises

        time.sleep(idx * 0.1)
        print(f'Lean4Worker id={self.idx} node={self.node} started.')
    
    def run(self, inputs, batched = True):
        # If (memory > threshold), wait until we have enough memory
        while psutil.virtual_memory().percent > MEMORY_THRESHOLD:
            print(f'Lean4Worker id={self.idx} node={self.node} waiting for memory...')
            time.sleep(5)

        if batched:
            tasks = dict(codes=[test_info['statement'] + '\n' + test_info['proof'] for test_info in inputs],
                        headers=[test_info.get('header', None) for test_info in inputs],
                        premises=False,
                        ast=False,
                        last_env=0)
            results = verify_lean4_file(**tasks)

            # get premises
            if self.collect_premises:
                for i, (test_info, result) in enumerate(zip(inputs, results)):
                    if result.get('complete', False):
                        task = dict(code=test_info['statement'] + '\n' + test_info['proof'],
                                    header=test_info.get('header', None),
                                    premises=True,
                                    ast=True,
                                    timeout=DEFAULT_TIMEOUT)
                        result = verify_lean4_file_premises(**task)
                        results[i] = result[0]
        else:
            assert len(inputs) == 1, "Single input only for premises mode"
            test_info = inputs[0]
            tasks = dict(code=test_info['statement'] + '\n' + test_info['proof'],
                        header=test_info.get('header', None),
                        premises=True,
                        ast=True,
                        timeout=DEFAULT_TIMEOUT)
            results = verify_lean4_file_premises(**tasks)

        outputs = []
        for test_info, result in zip(inputs, results):
            outputs.append(test_info | result)

        return outputs

def create_ray_lean4_actors(
        reserved_cpus: int = 0, 
        cpus_per_task: float = 4,
        **kwargs,
) -> List:
    for pg_id_str in ray.util.placement_group_table():
        pg_id_bin = PlacementGroupID(hex_to_binary(pg_id_str))
        pg = PlacementGroup(pg_id_bin)
        remove_placement_group(pg)

    print('Creating ray actors...')
    ray_workers = []
    
    for i, node in enumerate(ray.nodes()):
        ip = node['NodeManagerAddress']
        nr_cpus = int(node['Resources']['CPU']) - reserved_cpus
        nr_local_workers = int(nr_cpus / cpus_per_task)

        if nr_local_workers < 1:
            continue

        print(f'Creating {nr_local_workers} workers on node {ip}, host name {node["NodeManagerHostname"]}')
        pg = placement_group([{"CPU": nr_local_workers * cpus_per_task,
                               "node:" + ip: 0.1}], strategy="STRICT_PACK")
        ray.get(pg.ready())

        for j in range(nr_local_workers):
            worker = Lean4Worker.options(
                placement_group=pg,
                num_cpus=cpus_per_task,
            ).remote(i, j, **kwargs)
            ray_workers.append(worker)

    if not ray_workers:
        raise RuntimeError(
            f'No Lean workers fit in the LSF CPU allocation after reserving {reserved_cpus} CPUs.'
        )

    print(f'Ray actors created. Number of workers: {len(ray_workers)}')

    print('Initializing Lean4 environment...')
    warmup_path = os.path.join(DEFAULT_LEAN_WORKSPACE, '.lake/packages/REPL/test/aime_1983_p9.in')
    with open(warmup_path, 'r') as warmup_input:
        subprocess.run(
            [DEFAULT_LAKE_PATH, 'exe', 'repl'],
            cwd=DEFAULT_LEAN_WORKSPACE,
            stdin=warmup_input,
            stdout=subprocess.DEVNULL,
            check=True,
        )
    print('Lean4 environment initialized.')
    return ray_workers
