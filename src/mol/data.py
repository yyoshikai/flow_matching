from pathlib import Path

with open(Path(__file__).parent / "atoms.txt") as f:
    ATOMS = f.read().splitlines()

IDX2ATOM = {atom: i for i, atom in enumerate(ATOMS)}

