"""
Preprocessament de les dades d'entrenament i validació.

Llegeix els fitxers CSV originals (MZ + SMILES), genera el fingerprint Morgan
(ECFP4, radi=2, 2048 bits) per a cada molècula mitjançant RDKit, i converteix
els pics m/z en tokens enters (int(mz * 100)). El resultat es desa en format
.txt, una línia per molècula: bit0,bit1,...,bit2047;token1,token2,...

Per al conjunt de test, usar prepare_test_data.py.
"""
import re
import ast
import os
from rdkit import Chem
from rdkit.Chem import AllChem


def processar_csv_malformat(input_csv, output_txt):
    """
    Converteix un CSV (MZ + SMILES) al format .txt del model.

    El CSV té una estructura no estàndard: la columna de pics m/z conté una
    llista Python entre claudàtors (p.ex. [10.14, 200.5, ...]). La funció
    extreu aquesta llista amb regex i el SMILES com a darrer camp.

    Args:
        input_csv: Ruta al fitxer CSV d'entrada.
        output_txt: Ruta al fitxer .txt de sortida.
    """
    generador = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    resultats = []

    if not os.path.exists(input_csv):
        print(f"Error: No s'ha trobat el fitxer {input_csv}")
        return

    with open(input_csv, 'r', encoding='utf-8') as f:
        # Llegim totes les línies
        linies = f.readlines()

    print(f"Processant {len(linies)-1} línies...")

    for i, linia in enumerate(linies[1:], 1):
        try:
            # 1. Extreure la llista de MZ: busquem el que hi ha entre els claudàtors [ ]
            match_mz = re.search(r'\[.*\]', linia)
            if not match_mz:
                continue
            mz_str = match_mz.group(0)

            # 2. Extreure el SMILES: és l'últim element després de l'última coma
            # Netegem cometes i espais al final de la línia
            smiles = linia.strip().split(',')[-1].replace('"', '').strip()

            # 3. Generar Fingerprint (Morgan)
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            fp_bits = ",".join(list(generador.GetFingerprint(mol).ToBitString()))

            # 4. Processar MZ a Tokens (Enters)
            mz_list = ast.literal_eval(mz_str)
            # Convertim floats a enters truncats amb 2 decimals (x100), només positius
            tokens = sorted(list(set([
                str(int(float(mz) * 100))  #exemple: 10.145373 -> 1014
                for mz in mz_list
                if float(mz) >= 0
            ])))
            token_str = ",".join(tokens)

            # 5. Guardar format: label;token1,token2...
            if token_str:
                resultats.append(f"{fp_bits};{token_str}")

        except Exception:
            continue

    # Guardar el fitxer final per al model
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(resultats))
    
    print(f"Fet! S'han generat {len(resultats)} línies a {output_txt}")

if __name__ == "__main__":
    # TRAIN - dades per entrenar
    input_train = 'spec2finger/data/merge_final_train(in).csv'
    output_train = 'spec2finger/data/fingerMorgan_mz_train.txt'
    
    processar_csv_malformat(input_train, output_train)
    
    # VAL - dades per validar el model
    input_val = 'spec2finger/data/merge_final_validation(in).csv'
    output_val = 'spec2finger/data/fingerMorgan_mz_val.txt'
    
    processar_csv_malformat(input_val, output_val)

