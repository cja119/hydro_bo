# Sourced by the archetype PBS jobs. Edit for your cluster: modules, conda env, Gurobi licence.
module purge 2>/dev/null || true
# module load tools/prod Gurobi/<version>          # or however Gurobi is provided on your cluster
# export GRB_LICENSE_FILE=<path to your licence>    # if not set by the module
eval "$(conda shell.bash hook)"
conda activate hydro_bo
cd "$PBS_O_WORKDIR"                                 # submit from the repo root
export NCPUS="${NCPUS:-16}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # one thread per rollout process
