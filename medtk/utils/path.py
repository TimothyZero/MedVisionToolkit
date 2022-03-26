#  Copyright (c) 2020. The Medical Image Computing (MIC) Lab, 陶豪毅
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

import six

from .misc import is_str

if sys.version_info <= (3, 3):
    FileNotFoundError = IOError
else:
    FileNotFoundError = FileNotFoundError


def is_filepath(x):
    if is_str(x) or isinstance(x, Path):
        return True
    else:
        return False


def fopen(filepath, *args, **kwargs):
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not osp.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def _scandir_py35(dir_path, suffix=None):
    for entry in os.scandir(dir_path):
        if not entry.is_file():
            continue
        filename = entry.name
        if suffix is None:
            yield filename
        elif filename.endswith(suffix):
            yield filename


def _scandir_py(dir_path, suffix=None):
    for filename in os.listdir(dir_path):
        if not osp.isfile(osp.join(dir_path, filename)):
            continue
        if suffix is None:
            yield filename
        elif filename.endswith(suffix):
            yield filename


def scandir(dir_path, suffix=None):
    if suffix is not None and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')
    if sys.version_info >= (3, 5):
        return _scandir_py35(dir_path, suffix)
    else:
        return _scandir_py(dir_path, suffix)


def find_vcs_root(path, markers=('.git', )):
    """Finds the root directory (including itself) of specified markers.

    Args:
        path (str): Path of directory or file.
        markers (list[str], optional): List of file or directory names.

    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out


def get_git_hash(fallback='unknown', digits=None, package='medtk'):
    """Get the git hash of the current repo.

    Args:
        fallback (str, optional): The fallback string when git hash is
            unavailable. Defaults to 'unknown'.
        digits (int, optional): kept digits of the hash. Defaults to None,
            meaning all digits are kept.
        package: package to find

    Returns:
        str: Git commit hash.
    """

    if digits is not None and not isinstance(digits, int):
        raise TypeError('digits must be None or an integer')

    cwd = os.getcwd()
    try:
        import importlib
        os.chdir(find_vcs_root(importlib.__import__(package).__file__))
    except ModuleNotFoundError:
        print(f'Module {package} not found!')
        os.chdir(cwd)
        return None

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
        if digits is not None:
            sha = sha[:digits]
    except OSError:
        sha = fallback
    os.chdir(cwd)
    return sha