#!/usr/bin/env python3
#this goes in the ANTIPASTI/notebooks directory, run while in ANTIPASTI
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
from antipasti.utils.torch_utils import load_checkpoint
import subprocess
import numpy as np
import torch
import argparse
import pandas as pd

def predict(pdb_id):
    pdb = Path("notebooks/test_data/structure")/f"{pdb_id}.pdb"
    dccm = Path("notebooks/test_data/dccm_map")/f"{pdb_id}.npy"
    res = Path("notebooks/test_data/list_of_residues")/f"{pdb_id}.npy"
    if not res.exists():
        aa = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
        r = ['START-Ab']
        for l in open(pdb):
            if l.startswith('ATOM') and l[21] in ['H','L'] and l[17:20].strip() in aa:
                r.append(aa[l[17:20].strip()]+l[21:22])
        r = list(dict.fromkeys(r))
        np.save(res, r)
    if not dccm.exists():
        subprocess.run(['Rscript', 'scripts/pdb_to_dccm.r', str(pdb), str(dccm)])
    x = np.load(dccm)
    model = load_checkpoint("checkpoints/full_ags_all_modes/model_epochs_1044_modes_all_pool_1_filters_4_size_4.pt", x.shape[0])[0]
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(x).unsqueeze(0).unsqueeze(0))[0].item()
    return pred

def validate():
    df = pd.read_csv("data/sabdab_summary_all.tsv", sep='\t')
    df = df[['pdb', 'affinity']].drop_duplicates()
    df['log10_kd'] = -np.log10(df['affinity'].astype(float))
    results = []
    for _, row in df.iterrows():
        try:
            pred = predict(str(row['pdb']).lower())
            diff = abs(pred - row['log10_kd'])
            results.append({
                'pdb': row['pdb'],
                'predicted': pred,
                'actual': row['log10_kd'],
                'diff': diff,
                'pass': diff <= 0.5
            })
            print(f"PDB: {row['pdb']}")
            print(f"Predicted log10(KD): {pred:.2f}")
            print(f"Actual log10(KD): {row['log10_kd']:.2f}")
            print(f"Difference: {diff:.2f}")
            print(f"Pass: {diff <= 0.5}\n")
        except:
            continue
    passed = sum(1 for r in results if r['pass'])
    total = len(results)
    print(f"\nPassed {passed}/{total} tests ({passed/total*100:.1f}%)")
    return len(results) - passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pdb", help="PDB ID to test")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    args = parser.parse_args()
    if args.validate:
        sys.exit(validate())
    else:
        pred = predict(args.test_pdb)
        print(f"Predicted log10(KD): {pred:.2f}")
        print(f"Predicted KD: {10**pred:.2e} M")