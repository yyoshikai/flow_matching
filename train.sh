
export WORKDIR=/workspace

for lr in 1e-4 1e-5 3e-6 1e-6; do
    python train_mol.py --studyname lrs/$lr --lr $lr
done


# 2> >(sed -u "s|/workspace|/workspace/filesrv01/yoshikai|g" >&2) 2> >(sed -u "s|/workspace/filesrv01/yoshikai/ssd/2110/.venv|/workspace/filesrv01/yoshikai/envs/2110/.venv|g" >&2)