
export WORKDIR=/workspace
python train_mol.py 2> >(sed -u "s|/workspace|/workspace/filesrv01/yoshikai|g" >&2) 2> >(sed -u "s|/workspace/filesrv01/yoshikai/ssd/2110/.venv|/workspace/filesrv01/yoshikai/envs/2110/.venv|g" >&2)