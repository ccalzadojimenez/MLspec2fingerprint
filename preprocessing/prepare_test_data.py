"""
Converteix merge_final_test(in).csv al format del model:
  bit0,bit1,...,bit2047;token1,token2,...

Ús per defecte:
  input : data/merge_final_test(in).csv
  output: data/fingerMorgan_mz_test.txt
"""

from __future__ import annotations

import argparse
import ast
import os
import re
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem


def processar_csv_malformat(input_csv: str, output_txt: str) -> int:
    """Replica la mateixa lògica usada a train/val per generar labels+tokens."""
    generador = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    resultats: List[str] = []

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"No s'ha trobat el fitxer: {input_csv}")

    with open(input_csv, "r", encoding="utf-8") as f:
        linies = f.readlines()

    print(f"Processant {max(0, len(linies) - 1)} línies de test...")

    for i, linia in enumerate(linies[1:], 1):
        try:
            # 1) m/z entre claudàtors
            match_mz = re.search(r"\[.*\]", linia)
            if not match_mz:
                continue
            mz_str = match_mz.group(0)

            # 2) SMILES a l'últim camp
            smiles = linia.strip().split(",")[-1].replace('"', "").strip()

            # 3) fingerprint Morgan 2048 bits
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            fp_bits = ",".join(list(generador.GetFingerprint(mol).ToBitString()))

            # 4) tokens m/z = trunc(float*100), únics, positius, ordenats
            mz_list = ast.literal_eval(mz_str)
            tokens = sorted(
                list(
                    {
                        str(int(float(mz) * 100))
                        for mz in mz_list
                        if float(mz) >= 0
                    }
                )
            )
            token_str = ",".join(tokens)

            # 5) format final
            if token_str:
                resultats.append(f"{fp_bits};{token_str}")

        except Exception:
            # Mantenim el comportament tolerant del preprocess existent.
            continue

    out_dir = os.path.dirname(output_txt)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(resultats))

    print(f"Fet! Generades {len(resultats)} línies a: {output_txt}")
    return len(resultats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converteix CSV test a fingerMorgan_mz_test.txt."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/merge_final_test(in).csv",
        help="CSV d'entrada de test.",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default="data/fingerMorgan_mz_test.txt",
        help="TXT de sortida en format labels;tokens.",
    )
    args = parser.parse_args()

    processar_csv_malformat(args.input_csv, args.output_txt)


if __name__ == "__main__":
    main()
