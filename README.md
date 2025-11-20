# Max Flow Project for Parallel Algorithms Fall 2025

We investigate how effectively push-relabel method can be implemented on a GPU for sparse graphs. We design and implement a CuPy–based max–flow solver, and compare its performance to a serial CPU counterpart.

## Running the program

### 1. Create the virtual environment
```bash
python -m venv .venv
```

### 2. Install depedencies
```bash
pip install -r requirements.txt
```


### 3. Running the CuPy code
There are multiple versions of our CuPy implementation in the `src/` directory.

The script `run_cuda.py` invokes all three GPU implementations of the push-relabel as well as the baseline (iGraph).
This script is used mainly to verify correctness of the GPU implementation.
```bash
python run_cuda.py
```

The remaining run scripts are names as `run_<method>_<size>.py` and are intended to be invoked only by the batch scripts found in `slurm_scripts`.
This is because even on interactive sessions on TACC, 12K and larger nodes graphs can have prohibitively long runtimes.
