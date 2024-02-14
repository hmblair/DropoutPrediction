# utils.py

import xarray as xr
import ViennaRNA as vRNA
from tqdm import tqdm
import numpy as np
import pandas as pd

def vienna_fold(sequences : list[str]) -> tuple[list[str], list[float]]:
    """
    Fold a list of RNA sequences using ViennaRNA. The output is returned as a 
    list of dot-bracket structures. The free energy of each structure is also
    returned.
    """
    structures = []
    energies = []
    for sequence in tqdm(sequences):
        structure, energy = vRNA.fold(sequence)
        structures.append(
            structure
        )
        energies.append(
            energy
        )
    return structures, energies



def fasta_to_nc(file : str, fold : bool = True) -> None:
    """
    Convert a fasta file of RNA sequences to a netCDF file with the sequences 
    and their embeddings. The file is saved in the same directory as the fasta
    file with the same name and the extension .nc.

    Parameters
    ----------
    file : str
        Path to the fasta file.
    fold : bool
        Whether to fold the sequences using ViennaRNA. This is required if you
        will be using the LSTM + Structure model. 
    """
    # the embedding dictionary used by all the reads prediction models
    emb_dict = {
        'A' : 4,
        'C' : 5,
        'G' : 6,
        'U' : 7,
        'T' : 7,
        '.' : 8,
        '(' : 9,
        ')' : 10,
        '[' : 11,
        ']' : 12,
    }
    sequences = []
    with open(file, 'r') as f:
        current_seq = []
        for line in f:
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
            else:
                current_seq.append(line.strip())
    
    embedded_sequences = [
        [emb_dict[base] for base in seq] for seq in sequences
    ]

    if fold:
        structures = [
            vienna_fold(seq)[0] for seq in sequences
        ]
        embedded_structures = [
            [emb_dict[base] for base in struct] for struct in structures
        ]

        data = xr.Dataset(
            {
                'sequence' : (['batch'], sequences),
                'sequence_embeddings' : (['batch', 'nucleotide'], embedded_sequences),
                'structure' : (['batch'], structures),
                'structure_embeddings' : (['batch', 'nucleotide'], embedded_structures),
            }
        )

    else:
        data = xr.Dataset(
            {
                'sequence' : (['batch'], sequences),
                'sequence_embeddings' : (['batch', 'nucleotide'], embedded_sequences),
            }
        )

    # Save to netCDF
    data.to_netcdf(file.replace('.fasta', '.nc'))



def load_reads(
        experiment : str,
        model_names : str,
        dataset : str,
        renormalise: bool = True,
        ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if experiment not in ['2A3', 'DMS']: 
        raise ValueError('The experiment must be either 2A3 or DMS.')
    true_data = xr.load_dataset(f'data/{dataset}.nc')
    true_reads = true_data[f'reads_{experiment}'].values

    ix = 0 if experiment == '2A3' else 1
    predicted_reads = {}
    for name in model_names:
        loaded_data = xr.load_dataset(f'data/predictions/{name}/{dataset}.nc')
        reads = loaded_data['reads'].values[:, ix]
        if renormalise:
            reads = (reads[:, ix] / reads[:, ix].mean()) * 1000
            true_reads = (true_reads / true_reads.mean()) * 1000
        predicted_reads[name] = reads

    return true_reads, predicted_reads



def compute_metrics(
        true_reads : np.ndarray,
        predicted_reads : dict[str, np.ndarray],
        dataset : str,
        experiment : str,
        ) -> None:
    for name, reads in predicted_reads.items():
        rmsd = np.sqrt(np.mean((np.log10(reads) - np.log10(true_reads))**2))
        correlation = np.corrcoef(np.log10(reads), np.log10(true_reads))[0, 1]
        absolute_error = np.mean(np.abs(reads - true_reads))
        df_new = pd.DataFrame({
            'Model': [name],
            'RMSD': [rmsd],
            'Correlation': [correlation],
            'Mean Absolute Error': [absolute_error],
        }).round(
            {
                'RMSD': 3,
                'Correlation': 3,
                'Mean Absolute Error': 0,
            }
        )
        df = pd.concat([df, df_new], axis=0)
    df.to_csv(f'data/{dataset}_{experiment}.csv', index=False)