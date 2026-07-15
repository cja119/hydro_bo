# HYDRO-BO

Bayesian Optimisation of Hydrogen Export Infrastructure

## Installation

Run the following shell commands to clone onto your local system (reccomended):
```sh
# Install via git clone 
git clone https://github.com/cja119/hydro_bo.git

# Enter the repository
cd ./hydro_bo

# Install the package editable
pip install -e .
```

Run the following shell commands to clone onto your local system (not as robust):
```sh
# Install via git clone 
pip install git+https://github.com/cja119/hydro_bo.git
```

## Recreating Publication Results

Please note a licence for Gurobi is required.

Run scripts live under two directories. `scripts/idc/` holds the original
integrated-design-and-control problem; `scripts/parametric/` holds the
parametric (uncertain-parameter) work. Each is self-contained, with its own
`config.yml` and `tmp/` output tree.

### Run the Planning Optimisation

```sh
python scripts/idc/planning_model.py "NH3" # Or LH2
```

This generates a results file in `./scripts/idc/tmp/planning` that is then
accessed by your optimisation run. Both flavours share it — the parametric
config points at this directory via `general.planning_dir`.

## Run the Bayesian Optimisation

```sh
python scripts/idc/constrained_bo.py            # chance-constrained IDC run
python scripts/idc/unconstrained_bo.py          # unconstrained variant
```

Everything else is configured in `scripts/idc/config.yml`; the only CLI
overrides are `--vector` and `--ncpus`. Results are saved under
`./scripts/idc/tmp/<date>/<time>/<vector>/`.

## Run the Parametric (theta) Optimisation

```sh
python scripts/parametric/parametric_bo.py
```

The uncertain parameters are named in the `theta:` block of
`scripts/parametric/config.yml` and resolved against the catalog in
`hydro_bo.utils.theta`. The GP is fit over the joint space
`[x_design | theta]`, and each iteration optimises the design at a
Sobol-drawn theta node. Results are saved under
`./scripts/parametric/tmp/<date>/<time>/<vector>/`, with one `theta.<name>`
column per uncertain parameter.

## Visualising Trajectories

To visualise mpc trajectories, run:

```sh
python mpc.py "NH3" # Or LH2
```

## Acknowledgements

Read SOFTWARE.md to see software acknowledgements for this research. 
