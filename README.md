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

### Run the Planning Optimisation

Navigate to ./scripts/ and run the following:

```sh
python planning.py "NH3" # Or LH2
```

This will generate a results file in ./scripts/tmp/planning that will then be accessed by your optimisation run.

## Run the Bayesian Optimisation

```sh
python bayesopt.py "NH3" # Or LH2
```

This will save results in ./scripts/tmp/ray_results.

## Visualising Trajectories

To visualise mpc trajectories, run:

```sh
python mpc.py "NH3" # Or LH2
```

## Acknowledgements

Read SOFTWARE.md to see software acknowledgements for this research. 
