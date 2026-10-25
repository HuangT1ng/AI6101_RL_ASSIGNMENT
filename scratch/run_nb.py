"""Execute a notebook in place with nbclient (cwd = the notebook's folder, so figures/ is written next to it).
Usage: .venv/bin/python scratch/run_nb.py PATH.ipynb"""
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

path = pathlib.Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
start = time.time()
NotebookClient(nb, timeout=7200, kernel_name='python3', resources={'metadata': {'path': str(path.parent)}}).execute()
nbformat.write(nb, path)
errors = [i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and any(o.output_type == 'error' for o in c.outputs)]
print(f'executed {path.name} in {time.time() - start:.0f}s; cells with errors: {errors}')
