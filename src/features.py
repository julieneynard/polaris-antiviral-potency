"""Morgan fingerprint featurization (RDKit), fit-free and leak-free by construction.

Morgan fingerprints are a deterministic hash of molecular structure, so there is
nothing to "fit" on the training set here (unlike a scaler or feature selector) -
computing them on train and test separately introduces no leakage.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048


def smiles_to_morgan_fp(smiles: str, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_N_BITS) -> np.ndarray:
    """Convert a single SMILES/CXSMILES string to a Morgan fingerprint bit vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse molecule: {smiles!r}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(smiles_list: list[str], radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_N_BITS) -> np.ndarray:
    """Convert a list of SMILES strings to a Morgan fingerprint matrix."""
    return np.vstack([smiles_to_morgan_fp(s, radius=radius, n_bits=n_bits) for s in smiles_list])
