import gzip
import json
import logging
import os
import pickle
import shutil


def _read_file(filename):
    compressed = filename.endswith('.gz')
    base_filename = filename[:-3] if compressed else filename
    if not os.path.exists(filename):
        return None

    if compressed:
        with gzip.open(filename, 'rb') as file:
            file_data = file.read()
    else:
        with open(filename, 'rb' if base_filename.endswith('.pkl') else 'r') as file:
            file_data = file.read()

    if base_filename.endswith('.json'):
        return json.loads(file_data)
    if base_filename.endswith('.jsonl'):
        separator = '\n' if isinstance(file_data, str) else b'\n'
        return [json.loads(line) for line in file_data.split(separator) if line]
    if base_filename.endswith('.pkl'):
        return pickle.loads(file_data)
    return None


def read_file(filename):
    if filename is None:
        return None
    data = _read_file(filename)
    if data is None:
        data = _read_file(filename + '.gz')
    return data


def write_data(data, filename, content_type='json', no_compression=False):
    if no_compression:
        extension = ''
    else:
        extension = '.gz'
        if isinstance(data, str):
            data = data.encode()
        data = gzip.compress(data)

    full_filename = filename + extension
    os.makedirs(os.path.dirname(full_filename), exist_ok=True)
    mode = 'wb' if content_type == 'pickle' or extension else 'w'
    with open(full_filename, mode) as file:
        file.write(data)
    logging.debug(f'{full_filename} write complete')


def move_file(source, target):
    shutil.move(source, target)
    logging.debug(f'Moved {source} to {target}')


def cleanup_dir(directory):
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory, exist_ok=True)
    logging.debug(f'Deleted all files in {directory}')


def copy_dir(source, target):
    shutil.copytree(source, target, dirs_exist_ok=True)
    logging.debug(f'Copied {source} to {target}')


def path_exists(path):
    return path is not None and os.path.exists(path)
