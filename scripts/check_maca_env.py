import os
import sys
import traceback

print('Python:', sys.version)
print('Executable:', sys.executable)
print('PYTHONPATH:', os.environ.get('PYTHONPATH', ''))

errors = []

try:
    import torch
    print('Torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA runtime:', torch.version.cuda)
        print('GPU:', torch.cuda.get_device_name(0))
except Exception as e:
    errors.append(f'torch check failed: {e}')

try:
    import numpy as np
    import pandas as pd
    import pygame
    print('numpy:', np.__version__)
    print('pandas:', pd.__version__)
    print('pygame:', pygame.__version__)
except Exception as e:
    errors.append(f'base package check failed: {e}')

# project import check
try:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env_path = os.path.join(root, 'environment')
    if root not in sys.path:
        sys.path.insert(0, root)
    if env_path not in sys.path:
        sys.path.insert(0, env_path)
    import interface  # noqa: F401
    print('MaCA interface import: OK')
except Exception as e:
    errors.append(f'MaCA interface import failed: {e!r}')
    errors.append(traceback.format_exc())

if errors:
    print('\n[FAILED] environment check:')
    for e in errors:
        print('-', e)
    sys.exit(1)

print('\n[OK] environment is ready.')
