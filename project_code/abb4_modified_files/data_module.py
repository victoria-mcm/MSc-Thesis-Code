import tree, math, random, logging, warnings, dataclasses, os
import numpy as np
import pandas as pd

from typing import TypeVar, Optional, Iterator
T_co = TypeVar('T_co', covariant=True)

import ot as pot
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data.distributed import dist, DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from Bio import PDB
from Bio.PDB import Chain, Residue
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from abb4.data import utils as du
from abb4.data import parsers, errors, all_atom, residue_constants
from abb4.data.interpolant import _dataset_OT
from abb4.data import so3_utils
from openfold.data import data_transforms
from openfold.utils import rigid_utils

from anarcii import Anarcii


def heterogenos_length_collate(batch):
    """
    Custom collate function to handle batches where each element has different length.
    Pads all tensors with zeros to consistent length.
    """
    item_names = batch[0].keys()
    new_batch = {}
    for item_name in item_names:
        # Extract the values for each key from the batch
        item_values = [item[item_name] for item in batch]
        # Pad
        padded_item = pad_sequence(item_values, batch_first=True)
        new_batch[item_name] = padded_item
    
    return new_batch

def heterogenos_length_collate_ensemble(batch):
    """
    Custom collate function to handle batches where each element has different length.
    Use for ensemble batches with shape (B, N, seq_len, ...)
    Pads all tensors with zeros to consistent length.
    """
    item_names = batch[0].keys()
    ensemble_size = batch[0]['aatypes_1'].size(0)
    new_batch = {}
    for item_name in item_names:
        ensemble_values = []
        for ensemble_idx in range(ensemble_size):
            item_values = [item[item_name][ensemble_idx] for item in batch] # [B, seq_len, ...]
            # Pad
            padded_item = pad_sequence(item_values, batch_first=True) # [B, seq_len, ...]
            ensemble_values.append(padded_item)
        ensemble_values = torch.stack(ensemble_values, dim=0) # [N, B, seq_len, ...]
        new_batch[item_name] = ensemble_values.transpose(0, 1) # [B, N, seq_len, ...]

    return new_batch

def process_pickle(processed_file_path):
    '''Load and process a single row of the PDB CSV file.'''
    trans_1, rotmats_1, chain_feats, processed_feats = du.load_rigids(processed_file_path, return_feats=True)

    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f'Found NaNs in {processed_file_path}')
    atom14 = all_atom.atom37_to_atom14(chain_feats['all_atom_positions'], chain_feats)

    res_mask = torch.tensor(processed_feats['bb_mask']).int()
    imgt_numeric = torch.tensor(processed_feats['residue_index'])
    res_idx = torch.tensor(np.concatenate([np.arange((imgt_numeric < 1000).sum())+1,
                                            np.arange((imgt_numeric >= 1000).sum())+1001]))
    region = torch.tensor([du.imgt2region[imgt] for imgt in imgt_numeric])
    rmsd = torch.tensor(processed_feats['rmsd'].astype(float)) if 'rmsd' in processed_feats else None
    rmsf = torch.tensor(processed_feats['rmsf'].astype(float)).T if 'rmsf' in processed_feats else None  # (3, N) -> (N, 3) to match the shape of other feats

    chain_feats = {
        'aatypes_1': chain_feats['aatype'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'atom14_1': atom14,
        'atom14_atom_exists': chain_feats['atom14_atom_exists'],
        'atom37_atom_exists': chain_feats['atom37_atom_exists'],
        'residx_atom37_to_atom14': chain_feats['residx_atom37_to_atom14'],
        'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
        'torsion_angles': chain_feats['torsion_angles_sin_cos'],
        'torsion_mask': chain_feats['torsion_angles_mask'],
        'res_mask': res_mask,
        'res_idx': res_idx, # chain-wise sequential array; 
                            # starts at 1 (VH) and 1001 (VL), 
        'imgt_numeric': imgt_numeric, # sequential array of numeric part of IMGT numbering
                                        # VL numbers are offset by 1000
        'region': region, # numeric encoding of antibody regions FWH1, CDRH1, ..., FWL1, CDRL1, ...
    }

    if rmsd is not None:
        chain_feats['rmsd'] = rmsd
    if rmsf is not None:
        chain_feats['rmsf'] = rmsf

    return chain_feats

def _is_missing_seq(seq) -> bool:
    """True when a chain sequence is absent (NaN, None, or blank)."""
    if seq is None:
        return True
    if isinstance(seq, float) and pd.isna(seq):
        return True
    if isinstance(seq, str) and not seq.strip():
        return True
    return False

def _prepare_anarcii_input(vh_seq, vl_seq):
    has_heavy = not _is_missing_seq(vh_seq)
    has_light = not _is_missing_seq(vl_seq)
    if has_heavy and has_light:
        return [vh_seq, vl_seq], has_heavy, has_light
    if has_heavy:
        return [vh_seq], has_heavy, has_light
    if has_light:
        return [vl_seq], has_heavy, has_light
    raise ValueError('Both VH_seq and VL_seq are missing')

def _validate_anarcii_results(results, pdb_name, vh_seq, vl_seq):
    if results is None:
        raise ValueError(f'Anarcii returned None for {pdb_name}')

    seq1 = results.get('Sequence 1', {})
    if seq1.get('numbering') is None:
        err = seq1.get('error', 'unknown numbering failure')
        raise ValueError(
            f'Anarcii failed to number {pdb_name}: {err}. VH={vh_seq}, VL={vl_seq}'
        )

    if 'Sequence 2' in results and results['Sequence 2'].get('numbering') is None:
        err = results['Sequence 2'].get('error', 'unknown numbering failure')
        raise ValueError(
            f'Anarcii failed to number light chain for {pdb_name}: {err}. VL={vl_seq}'
        )

class SingleStructureDataset(Dataset):
    """Training-like dataset to be used for training, as well as
    training-like validation and testing (where we compute losses 
    against ground thruth, rather than assessing de-novo generations
    without reference to a ground truth). Unsuitable for inference.

    Administers a pre-filtered csv file (see pre_filter_ab_data.py) 
    with paths to pre-processed, pickled dictionaries with antibody
    features derived from PDBs (see process_ab_pdb_files.py).
    """
    def __init__(self, *, cfg, pdb_csv):
        self._log = logging.getLogger(__name__)
        self.pdb_csv = pdb_csv

        self._samples_cfg = cfg
        self.gen_extent = cfg.gen_extent
        self.gen_regions = cfg.gen_regions
        self.proba_sampling = cfg.sim_proba_sampling
        
        self._igso3 = None
        
        self._ot_cfg = cfg.ot
        if self._ot_cfg.ot_fn == "emd":
            self.ot_fn = pot.emd
        else:
            raise ValueError(f'OT function {self._ot_cfg.ot_fn} not supported yet')

    @property
    def igso3(self):
        # isotropic Gaussian-equivalent on SO3 
        # (=truncated-series form of closed-form solution to Wiener diffusion process on SO3)
        # see FrameDiff paper, Appendix E
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def __len__(self):
        if self.proba_sampling:
            return self.pdb_csv.simulation_ID.nunique() # one structure per simulation
        else:
            return len(self.pdb_csv) # all structures

    def __getitem__(self, idx):
        if self.proba_sampling:
            # index one simulation ID from pdb csv
            sim_ID = self.pdb_csv['simulation_ID'].unique()[idx]
            # select all frames from that simulation
            sim_frames = self.pdb_csv[self.pdb_csv['simulation_ID'] == sim_ID]
            # sample one frame based on its probability
            if self._samples_cfg.sim_proba_sampling_mode == 'sample':
                csv_row = sim_frames.sample(n=1, weights='joint_proba_norm').squeeze()
            elif self._samples_cfg.sim_proba_sampling_mode == 'highest':
                csv_row = sim_frames.nlargest(1, 'joint_proba_norm').squeeze()
            else:
                raise ValueError(f'Unknown simulation probability sampling mode')
        else:
            # index one structure from pdb csv
            csv_row = self.pdb_csv.iloc[idx]

        chain_feats = process_pickle(csv_row['processed_path'])
        # chain_feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx
        chain_feats['pdb_name_enc'] = torch.ones(1, dtype=torch.long) * int(csv_row['pdb_name_enc'])
        
        # precalculate trans_0 and rotmats_0 if OT mode is 'dataset'
        # saves time during training step
        if self._ot_cfg.use_ot and self._ot_cfg.ot_mode == 'dataset':
            chain_feats = {k: v[None,...] for k, v in chain_feats.items()}
        
            trans_0, rotmats_0 = _dataset_OT(
                chain_feats,
                self.pdb_csv,
                self.ot_fn,
                self._ot_cfg,
                self.igso3,
                self._samples_cfg.rots_dist,
                device=chain_feats['res_mask'].device
                )
            chain_feats['trans_0'] = trans_0
            chain_feats['rotmats_0'] = rotmats_0
            chain_feats = {k: v.squeeze(0) for k, v in chain_feats.items()}
        # TODO: probably move this functionality into separate function
        if self.gen_extent == 'full':
            chain_feats['diffuse_mask'] = chain_feats['res_mask']
        else:
            raise NotImplementedError(
                f'Partial generation extent not yet implemented.')
            assert self.gen_region is not None, 'For partial generation, specify gen_regions whose generation is to be trained.'
            # TODO: check how this is done in scaffolding paper and implement
            # TODO: determine diffuse mask based on anarci numbering and north definitions provided in csv_row
            # create diffuse mask according to user-specified gen_regions
            # if succesful, check that no MASKs/UNKNOWNS/i.e. 'X' AAs are present OUTSIDE the diffuse mask
        return chain_feats

class EnsembleDataset(Dataset):
    """Dataset used for ensmble training and validation.
    The dataset loads a set of N structures from the same simulation.
    Otherwise this is identical to the SingleStructureDataset.

    Recommended to set N to the training batch size and batch size to 1.
    Optimal transport is not performed here as ensemble validation metrics
    are computed with the full distance matrix of all predicted and gt structures.
    """
    def __init__(self, *, cfg, pdb_csv):
        self._log = logging.getLogger(__name__)
        self.pdb_csv = pdb_csv

        self._samples_cfg = cfg
        self.gen_extent = cfg.gen_extent
        self.gen_regions = cfg.gen_regions
        self.ensemble_size = cfg.ensemble_size
        self.proba_sampling = cfg.sim_proba_sampling
        
        self._igso3 = None
        
        self._ot_cfg = cfg.ot
        if self._ot_cfg.ot_fn == "emd":
            self.ot_fn = pot.emd
        else:
            raise ValueError(f'OT function {self._ot_cfg.ot_fn} not supported yet')

    @property
    def igso3(self):
        # isotropic Gaussian-equivalent on SO3 
        # (=truncated-series form of closed-form solution to Wiener diffusion process on SO3)
        # see FrameDiff paper, Appendix E
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def __len__(self):
        # only load one data point per unique sequence
        return self.pdb_csv.simulation_ID.nunique()

    def __getitem__(self, idx):
        chain_feats_ensemble = []

        # index one simulation ID from pdb csv
        sim_ID = self.pdb_csv['simulation_ID'].unique()[idx]
        # select all frames from that simulation
        sim_frames = self.pdb_csv[self.pdb_csv['simulation_ID'] == sim_ID]
        # sample N frames from the same simulation based on their probability
        if self.proba_sampling:
            sampled_frame_rows = sim_frames.sample(
                n=self.ensemble_size,
                weights='joint_proba_norm',
                replace=True
            )
        else:
            sampled_frame_rows = sim_frames.sample(
                n=self.ensemble_size,
                replace=False
            )

        for _, row in sampled_frame_rows.iterrows():
            chain_feats = process_pickle(row['processed_path'])
            chain_feats['pdb_name_enc'] = torch.ones(1, dtype=torch.long) * int(row['pdb_name_enc'])

            # precalculate trans_0 and rotmats_0 if OT mode is 'dataset'
            if self._ot_cfg.use_ot and self._ot_cfg.ot_mode == 'dataset':
                chain_feats = {k: v[None,...] for k, v in chain_feats.items()}
            
                trans_0, rotmats_0 = _dataset_OT(
                    chain_feats,
                    self.pdb_csv,
                    self.ot_fn,
                    self._ot_cfg,
                    self.igso3,
                    self._samples_cfg.rots_dist,
                    device=chain_feats['res_mask'].device
                    )
                chain_feats['trans_0'] = trans_0
                chain_feats['rotmats_0'] = rotmats_0
                chain_feats = {k: v.squeeze(0) for k, v in chain_feats.items()}

            if self.gen_extent == 'full':
                chain_feats['diffuse_mask'] = chain_feats['res_mask']
            else:
                raise NotImplementedError(
                    f'Partial generation extent not yet implemented.')
            chain_feats_ensemble.append(chain_feats)

        chain_feats_ensemble = {key: torch.stack([sample[key] for sample in chain_feats_ensemble]) for key in chain_feats_ensemble[0]}
        return chain_feats_ensemble

class PredictDataset(Dataset):
    """Dataset used for model inference"""

    def __init__(self, *, cfg, pdb_csv):
        self._log = logging.getLogger(__name__)

        self._samples_cfg = cfg

        self.numbering_model = Anarcii(
            seq_type="antibody",
            batch_size=32, 
            cpu=True,
            ncpu=8, 
            mode="accuracy",
            verbose=False
        )

        keep_indices = []
        for idx, row in pdb_csv.iterrows():
            try:
                seq, _, _ = _prepare_anarcii_input(row.VH_seq, row.VL_seq)
                results = self.numbering_model.number(seq)
                _validate_anarcii_results(
                    results, row.pdb_name, row.VH_seq, row.VL_seq
                )
                keep_indices.append(idx)
            except ValueError as exc:
                self._log.warning('Skipping unnumberable entry: %s', exc)

        pdb_csv = pdb_csv.loc[keep_indices].reset_index(drop=True)
        if pdb_csv.empty:
            raise ValueError('No numberable sequences remain in predict CSV')

        n = self._samples_cfg.num_samples
        self.pdb_csv = pdb_csv.loc[pdb_csv.index.repeat(n)].reset_index(drop=True)
        self.pdb_csv['copy_index'] = self.pdb_csv.groupby(self.pdb_csv.pdb_name).cumcount()

    def __len__(self):
        return len(self.pdb_csv)

    def process_feats(self, csv_row):

        chain_feats = {}

        vh_seq = csv_row.VH_seq
        vl_seq = csv_row.VL_seq
        seq, has_heavy, has_light = _prepare_anarcii_input(vh_seq, vl_seq)

        results = self.numbering_model.number(seq)
        _validate_anarcii_results(results, csv_row.pdb_name, vh_seq, vl_seq)

        imgt_numeric = []
        Fv_seq = []
        for numb in results['Sequence 1']['numbering']:
            if numb[1] != '-':
                imgt_offset = 1000 if (has_light and not has_heavy) else 0
                imgt_numeric.append(numb[0][0] + imgt_offset)
                Fv_seq.append(numb[1])

        if has_heavy and has_light and 'Sequence 2' in results:
            for numb in results['Sequence 2']['numbering']:
                if numb[1] != '-':
                    imgt_numeric.append(numb[0][0]+1000)
                    Fv_seq.append(numb[1])

        imgt_numeric = np.array(imgt_numeric, dtype=int)

        chain_feats['aatypes_1'] = torch.tensor([residue_constants.restype_order[aa] for aa in Fv_seq])
        chain_feats['aatype'] = chain_feats['aatypes_1']
        chain_feats['imgt_numeric'] = torch.tensor(imgt_numeric)
        chain_feats['region'] = torch.tensor([du.imgt2region[imgt] for imgt in imgt_numeric])
        chain_feats['res_idx'] = torch.tensor(np.concatenate([np.arange((imgt_numeric < 1000).sum(), dtype=int)+1,
                                                np.arange((imgt_numeric >= 1000).sum(), dtype=int)+1001]))

        chain_feats['res_mask'] = torch.ones_like(chain_feats['aatypes_1'])
        chain_feats['diffuse_mask'] = torch.ones_like(chain_feats['aatypes_1'])

        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats.pop('residx_atom14_to_atom37')
        chain_feats.pop('atom14_atom_exists')
        chain_feats.pop('aatype')

        return chain_feats

    def __getitem__(self, idx):
        # return one structure from pdb csv
        csv_row = self.pdb_csv.iloc[idx]

        chain_feats = self.process_feats(csv_row)

        chain_feats['pdb_name'] = torch.tensor(
            list(bytearray(csv_row['pdb_name'], "utf-8")), dtype=torch.uint8
        )
        chain_feats['copy_index'] = torch.ones(1, dtype=torch.int16) * int(csv_row['copy_index'])

        return chain_feats

class LengthDataset(Dataset):
    """
    Inference-only dataset for full-extent, unconditional antibody generation only. 
    Unsuitable for training, partial, or conditional inference.

    The dataset is constructed by first applying hard length filters to a reference dataset,
    then sampling chain lengths independently from the middle 95% of the respective empirical
    distributions in the filtered reference set. Unique chain-length combinations are sampled 
    until the pre-set number of combinations is reached. Appropriate residue indices and 
    diffusion masks are then constructed for each sample.
    
    Config Flags:
    - samples_cfg.seed: Random seed for reproducibility.
    - samples_cfg.csv_path: Path to the reference dataset CSV file.
    - samples_cfg.min_length: Minimum total antibody length (VH+VL).
    - samples_cfg.max_length: Maximum total antibody length (VH+VL).
    - samples_cfg.min_length_heavy: Minimum length of the VH chain.
    - samples_cfg.max_length_heavy: Maximum length of the VH chain.
    - samples_cfg.min_length_light: Minimum length of the VL chain.
    - samples_cfg.max_length_light: Maximum length of the VL chain.
    - samples_cfg.num_combos: Number of unique VH/VL-length combinations to sample.
    - samples_cfg.num_samples_per_combo: Number of samples to generate per unique combination.
    """

    def __init__(self, cfg):
        self.data_type = 'lens'

        self._samples_cfg = cfg.samples
        random.seed(self._samples_cfg.seed)
        
        # read in reference dataset on which to base sampling distribution
        df = pd.read_csv(self._samples_cfg.csv_path)
        seqs = df['full_seq'].str.split('/', expand=True)
        df['len_h'], df['len_l'] = seqs[0].str.len(), seqs[1].str.len()
        df['len_tot'] = df['len_h'] + df['len_l']
        # apply hard filters
        df = df[(df['len_tot'] <= self._samples_cfg.max_length) & (df['len_tot'] >= self._samples_cfg.min_length)]
        df = df[(df['len_h'] <= self._samples_cfg.max_length_heavy) & (df['len_h'] >= self._samples_cfg.min_length_heavy)]
        df = df[(df['len_l'] <= self._samples_cfg.max_length_light) & (df['len_l'] >= self._samples_cfg.min_length_light)]

        # reconstruct middle 95% of VH and VL length distros from reference dataset
        lengths_h, probabilities_h = self.get_valid_lens_probs(df, 'len_h')
        lengths_l, probabilities_l = self.get_valid_lens_probs(df, 'len_l')

        # sample a set number of unique VH/VL-length combinations
        combos = set()
        while len(combos) < self._samples_cfg.num_combos:
            h = np.random.choice(lengths_h, p=probabilities_h)
            l = np.random.choice(lengths_l, p=probabilities_l)
            if h+l >= self._samples_cfg.min_length and h+l <= self._samples_cfg.max_length and (h,l) not in combos:
                combos.add((h,l))   
        
        # set up res_idx and full-antibody masks for each sample
        sample_id, rows = -1, []
        for len_h, len_l in combos:
            for i in range(self._samples_cfg.num_samples_per_item):
                idx_h = torch.arange(len_h) + 1
                idx_l = torch.arange(len_l) + 1001
                res_idx = torch.cat([idx_h, idx_l])
                res_mask = torch.ones(len_h + len_l).int()
                diffuse_mask = res_mask
                sample_id += 1
                sample = {'sample_id': sample_id,
                          'res_idx': res_idx,
                          'res_mask': res_mask,
                          'diffuse_mask': diffuse_mask}
                seq_len = len_h + len_l
                row = {'seq_len': seq_len, 'features': sample}
                rows.append(row)
        self.samples_df = pd.DataFrame(rows)

    def get_valid_lens_probs(self, df, column='len_h'):
        """ Take lengths of individual chain (VH or VL) in reference dataset.
        Return lengths within the middle 95% of that distribution and their probabilities. """
        lengths, counts = np.unique(df[column], return_counts=True)
        probabilities = counts / counts.sum()

        cumulative_probabilities = np.cumsum(probabilities)
        valid_mask = (cumulative_probabilities >= 0.025) & (cumulative_probabilities <= 0.975)

        lengths = lengths[valid_mask]
        probabilities = probabilities[valid_mask]/probabilities[valid_mask].sum()

        return lengths, probabilities

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        # return a single sample entry
        return self.samples_df.iloc[idx]['features']
    
class FastaDataset(Dataset):
    """Inference-only dataset of sequences for full-extent structure generation 
    (structure prediction/forward folding). Unsuitable for training or partial-extent
    inference. Administers a fasta-like file with antibody sequences in the following 
    format:

    >unique_seq_id
    VHSEQVENCE/VLSEQVENCE

    Internally constructs features for each sequence, and stores them in a DataFrame
    with columns 'seq_id', 'seq', 'seq_len', 'sample_id', and 'features'.
    The 'features' column is dictionary-valued.
    """
    def __init__(self, cfg):
        self.data_type = 'seqs'

        self._samples_cfg = cfg.samples
        random.seed(self._samples_cfg.seed)
        self.samples_df = self.set_up_dataframe(self._samples_cfg.fasta_path)

    def read_fasta(self, fasta_path):
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        seqs = []
        for i, line in enumerate(lines):
            if line.startswith('>'):
                seq_id = line[1:].strip()
                seq = lines[i+1].strip()
                seqs.append((seq_id, seq))
        seqs_df = pd.DataFrame(seqs, columns=['seq_id', 'seq'])
        return seqs_df

    def create_features(self, row):
        VH_seq, VL_seq = row['seq'].split('/')
        len_h, len_l = len(VH_seq), len(VL_seq)
        idx_h = torch.arange(len_h) + 1
        idx_l = torch.arange(len_l) + 1001

        res_idx = torch.cat([idx_h, idx_l])
        res_mask = torch.ones(len_h + len_l).int()
        diffuse_mask = res_mask
        aatypes_1 = torch.cat([ torch.tensor(du.seq_to_aatype[VH_seq]),
                                torch.tensor(du.seq_to_aatype[VL_seq])])
        
        return {'res_idx': res_idx,
                'res_mask': res_mask,
                'diffuse_mask': diffuse_mask,
                'aatypes_1': aatypes_1}
    
    def set_up_dataframe(self, fasta_path):
        # read file and create dataframe w/ cols 'seq_id', 'seq', 'seq_len' 
        data_csv = self.read_fasta(fasta_path)
        data_csv['seq_len'] = data_csv['seq'].str.len()
        
        # create features for each unique sequence
        data_csv['features'] = data_csv.apply(self.create_features, axis=1)

        # replicate each row #samples_per_sequence times
        data_csv = data_csv.loc[data_csv.index.repeat(self._samples_cfg.num_samples_per_item)].reset_index(drop=True)

        # define unique sample_ids for each row and add it to features
        data_csv['sample_id'] = data_csv.groupby('seq_id').cumcount().astype(str)
        data_csv['sample_id'] = data_csv['seq_id'] + '_' + data_csv['sample_id']
        data_csv['features'] = data_csv.apply(lambda row: {**row['features'], 'sample_id': row['sample_id']}, axis=1)

        return data_csv      

    def __len__(self):
        return len(self.samples_df)
    
    def __getitem__(self, idx):
        # return a single sequence entry
        return self.samples_df.iloc[idx]['features']

class PdbDataset(Dataset):
    """Inference-only dataset of PDB structures to be used for 
    full-extent sequence generation (inverse folding) or partial-extent
    inference (forward/inverse folding and sequence-structure co-generation)."""
    
    def __init__(self, cfg):
        self.data_type = 'pdbs'

        self.gen_extent = cfg.gen_extent
        self.gen_regions = cfg.gen_regions

        self._samples_cfg = cfg.samples
        pdb_dir = self._samples_cfg.pdb_dir
        random.seed(self._samples_cfg.seed)
    
        # you can specify diffuse_masks in a csv file
        if self.gen_regions == 'mask' and os.path.exists(os.path.join(pdb_dir, 'pdb_masks.csv')):
            # should have columns 'pdb_id', 'diffuse_mask'
            self._masks_csv = pd.read_csv(os.path.join(pdb_dir, 'pdb_masks.csv'))
        else:
            self._masks_csv = None

        pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]

        self.pdb_parser = PDB.PDBParser(QUIET=True)
        self.samples_df = self.setup_dataframe(pdb_files)

    def check_pdb_struc(self, pdb_struc, pdb_path):
        # count chains
        struct_chains = {chain.id.upper() : chain for chain in pdb_struc.get_chains()}
        num_chains = len(struct_chains)
        if num_chains != 2:
            raise errors.DataError(f'Expected 2 chains, got {num_chains} in file {pdb_path}.')
    
        # ensure chains have correct IDs
        if 'H' not in struct_chains or 'L' not in struct_chains:
            raise errors.DataError(f'Expected chains H and L, got {list(struct_chains.keys())} in file {pdb_path}.')

        return struct_chains
    
    def re_index_and_extract_seq(self, struct_chains):
        # re-index residues starting at 1 and 1001 for VH, VL, respectively
        # also get sequences as strings
        seqs = {}
        for chain_id in ['H', 'L']:
            seq = []
            chain = struct_chains[chain_id]
            new_chain = Chain.Chain(chain_id)
            for i, res in enumerate(chain, start=1):
                
                try:
                    one_letter_code = PDB.Polypeptide.index_to_one(
                                        PDB.Polypeptide.three_to_index(res.resname)
                                    )
                except KeyError:
                    one_letter_code = 'X'
                seq.append(one_letter_code)

                if chain_id == "H":
                    new_id = (res.id[0], i, res.id[2])
                else:
                    new_id = (res.id[0], i+1000, res.id[2])
                new_res = Residue.Residue(new_id, res.resname, res.segid)
                for atom in res:
                    new_res.add(atom)
                new_chain.add(new_res)
            struct_chains[chain_id] = new_chain
            seqs[chain_id] = ''.join(seq)
        
        # merge VH, VL into a single chain with discontinuous index
        merged_chain = Chain.Chain("M")
        for chain in [struct_chains['H'], struct_chains['L']]:
            for res in chain:
                merged_chain.add(res)

        return {"M": merged_chain}, seqs

    def extract_merged_chain_features(self, struct_chains):
        assert len(struct_chains) == 1, f'Expected 1 merged chain, got {len(struct_chains)} chains.'
        
        chain_id, chain = next(iter(struct_chains.items()))
        chain_id = du.chain_str_to_int(chain_id) # ascii index of 'M'
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        # run through OpenFold data transforms
        chain_feats = { 'aatype': torch.tensor(chain_dict['aatype']).long(),
                        'all_atom_positions': torch.tensor(chain_dict['atom_positions']).double(),  # centred coords
                        'all_atom_mask': torch.tensor(chain_dict['atom_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
                            chain_feats['rigidgroups_gt_frames'])[:, 0]

        return {'aatypes_1': chain_feats['aatype'],
                'rotmats_1': rigids_1.get_rots().get_rot_mats(),
                'trans_1':   rigids_1.get_trans(),
                'res_mask':  torch.tensor(chain_dict['bb_mask']).int(),
                'res_idx':   torch.tensor(chain_dict['residue_index']) }

    def process_pdb(self, pdb_path):
        pdb_id = pdb_path.split('/')[-1].split('.')[0]
        warnings.simplefilter('ignore', PDBConstructionWarning)
        structure = self.pdb_parser.get_structure(pdb_id, pdb_path) # read pdb file
        struct_chains = self.check_pdb_struc(structure, pdb_path)   # sanity-check structure
        merged_chains, seqs = self.re_index_and_extract_seq(struct_chains)

        seqlen = len(seqs['H']) + len(seqs['L'])
        features = self.extract_merged_chain_features(merged_chains)

        if self._masks_csv is not None:
            diffuse_mask = self._masks_csv[self._masks_csv['pdb_id'] == pdb_id]['diffuse_mask'].values[0]
            features['diffuse_mask'] = torch.tensor(diffuse_mask).int()
        elif self.gen_extent == 'full':
            features['diffuse_mask'] = features['res_mask']
        else:
            unique_aatypes = torch.unique(features['aatypes_1'])
            if len(unique_aatypes) <= 5:
                raise ValueError(
                    f'The sequence in {pdb_path} seems to be nonsense, but you have '
                    'specified partial generation extent. Did you mean to set '
                    'gen_extent to "full"?'
                )
            # TODO: determine diffuse mask and probably move this into its own function
            # run seqs anarci to get CDRs by North definition, raise error if innumerable
            # create diffuse mask according to user-specified gen_regions
            # if succesful, check that no MASKs/UNKNOWNS/i.e. 'X' AAs are present OUTSIDE the diffuse mask
            raise NotImplementedError(f'Partial generation extent not yet implemented, unless masks specified in file.')

        return {'seq_len': seqlen, 'features': features}

    def setup_dataframe(self, pdb_files):
        rows = []
        for pdb_path in pdb_files:
            row = self.process_pdb(pdb_path)
            for i in range(self._samples_cfg.num_samples_per_item):
                sample_id = f'{pdb_path.split("/")[-1].split(".")[0]}_{i}'
                row['features']['sample_id'] = sample_id
                rows.append(row)
        return pd.DataFrame(rows) # columns: 'seq_len', 'features'
    
    def __len__(self):
        return len(self.samples_df)
    
    def __getitem__(self, idx):
        # return a single entry
        return self.samples_df.iloc[idx]['features']

# TODO: fix this mess
class BatchSamplerWrapper(BatchSampler):
    ''' Wrapper for DistributedInferenceBatchSampler, because PyTorch Lightning
    insists the predict-mode dataloaders have a PyTorch BatchSampler API interface.
    '''

    def __init__(self, sampler, batch_size=1, drop_last=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        # print('WRAPPED SAMPLER:', self.sampler, 'BATCH SIZE:', self.batch_size)

    def __iter__(self):
        for batch in self.sampler:
            # print(f'BATCH: {batch}')
            yield [batch]   # monkeypatch to circumvent batchsampler injection problems in Lightning
                            # only works with batch_size=1

    def __len__(self):
        return len(self.sampler)

class DistributedInferenceBatchSampler(DistributedSampler):

    def __init__(self, dataset: Dataset,
                       batch_size: int = 1,
                       num_replicas: Optional[int] = None,
                       rank: Optional[int] = None, 
                       shuffle: bool = False,               # required by Lightning API for predict dataloaders, not used
                       seed: int = 0,                       # required by Lightning API for predict dataloaders, not used
                       drop_last: bool = False,             # required by Lightning API for predict dataloaders, not used
                       
        ) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, 
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self._log = logging.getLogger(__name__)
        
        self.batch_size = batch_size
        self.make_batches_for_replica()

    def assign_batch_ids_csv(self, group):
        group_size = len(group)
        num_batches = (group_size + self.batch_size - 1) // self.batch_size
        batch_ids = list(np.repeat(np.arange(num_batches), self.batch_size)[:group_size])
        group['batch_id'] = [f'{group.name}_{i}' for i in batch_ids]
        return group
    
    def create_batches_by_seq_len(self, data_csv):
        """data_csv: pandas DataFrame with columns 'seq_len' and 'features'."""
        # assign batch ids to each row
        data_csv = data_csv.groupby('seq_len', group_keys=False).apply(self.assign_batch_ids_csv)

        # create batchlist
        batches = [] 
        for _, group in data_csv.groupby('batch_id'):
            batch = group['index'].tolist()
            batches.append(batch)
    
        # [[idx1, idx2, ...], [idxN, idxM, ...], ...]
        return batches

    def make_batches_for_replica(self):
        dataset = self.dataset.samples_df
        dataset['index'] = list(range(len(dataset))) # needed later

        # [[idx1, idx2, ...], [idxN, idxM, ...], ...]
        batches = self.create_batches_by_seq_len(dataset)

        # distribute batches across replicas by subsampling
        # every nth batch for current replica
        replica_batches = batches[self.rank::self.num_replicas]  
        self.batches = replica_batches

    def __iter__(self) -> Iterator[T_co]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

def subsample_one_per_cluster(pdb_csv, seed):
    """Return one row per cluster_ids, preserving original dataset indices."""
    df = pdb_csv.copy()
    df['index'] = list(range(len(df)))
    return df.groupby('cluster_ids').sample(n=1, random_state=seed).reset_index(drop=True)


class DistributedTrainBatchSampler(DistributedSampler):

    def __init__(self, dataset: Dataset,
                       bsampler_cfg=None,
                       num_replicas: Optional[int] = None,
                       rank: Optional[int] = None, 
                       shuffle: bool = True,
                       seed: int = 0, 
                       drop_last: bool = False
        ) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle,
                         seed=seed, drop_last=drop_last)
        self._log = logging.getLogger(__name__)
        
        self.cfg = bsampler_cfg
        self.batch_size = self.cfg.batch_size

        self.make_batches_for_epoch_and_replica() # creates self.batches = [[idx1, idx2, ...], [idxN, idxM, ...], ...]
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}.\n\
                         Contains {len(self.batches)} batches of size {self.batch_size}.')

    def make_batches_for_epoch_and_replica(self):
        # start from complete dataset (train, val or test)
        full_dataset = self.dataset.pdb_csv
        full_dataset['index'] = list(range(len(full_dataset))) # needed later
        
        # subsample one pdb per sequence-similarity cluster (for this epoch)
        seed = self.seed + self.epoch # unique to each epoch, common across replicas
        full_dataset = subsample_one_per_cluster(full_dataset, seed=seed)

        # shuffle pdb indices, if specified
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            indices = torch.randperm(len(full_dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            # indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
            indices = list(range(len(full_dataset)))  # type: ignore[arg-type]
        # also subsample pdb indices if specified
        if self.cfg.num_struc_samples is not None:
            indices = indices[:self.cfg.num_struc_samples]

        # apply shuffling and subsampling
        shuffled_dataset = full_dataset.iloc[indices]

        # create length-homogenous batches of equal size batch_size
        batches = []
        for seq_len, len_df in shuffled_dataset.groupby('modeled_seq_len'):
            
            # num of *complete* batches for *this* sequence-length
            # this will drop some structures in the incomplete last batch 
            # (for this epoch)
            num_batches = math.floor(len(len_df) / self.batch_size)

            for i in range(num_batches):
                batch_df = len_df.iloc[i*self.batch_size:(i+1)*self.batch_size]
                batch_indices = batch_df['index'].tolist()
                batches.append(batch_indices)

        # shuffle batches to mix lengths
        new_order = torch.randperm(len(batches), generator=g).tolist()
        batches = [batches[i] for i in new_order]
        total_num_batches = len(batches)
        assert total_num_batches > 0, 'No batches created.'
        
        # make sure the number of batches is divisible by num_replicas
        num_augments = -1
        while total_num_batches % self.num_replicas != 0:
            if not self.drop_last:
                total_num_batches += 1
                num_augments += 1
            else:
                total_num_batches -= 1
                num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if total_num_batches >= len(batches):
            padding_size = total_num_batches - len(batches)
            batches += batches[:padding_size]
        else:
            batches = batches[:total_num_batches]
        assert len(batches) == total_num_batches, f'len(batches)={len(batches)} != total_num_batches={total_num_batches}'
        
        # distribute batches across replicas by subsampling
        # every nth batch for current replica (and .)
        replica_batches = batches[self.rank::self.num_replicas]  
        self.batches = replica_batches
        assert len(self.batches) > 0, 'No batches created.'
        for batch in self.batches:
            assert len(batch) == self.batch_size, f"Batch length is {len(batch)}, but expected length is {self.batch_size}."

    def __iter__(self) -> Iterator[T_co]:
        if self.epoch > 0:
            self.make_batches_for_epoch_and_replica()
        self.epoch += 1
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)

class DataModule(LightningDataModule):
    
    def __init__(self, module_cfg):
        super().__init__()
        self._log = logging.getLogger(__name__)
        self.module_cfg = module_cfg
        
        self.task = module_cfg.task
        self.gen_extent = module_cfg.gen_extent

    def setup(self, stage: str):
        # FOR INFERENCE
        if self.module_cfg.module.inference_only_mode:
            pdb_csv = pd.read_csv(self.module_cfg.dataset.predict.csv_path) 
            self.preds = PredictDataset(
                cfg=self.module_cfg.dataset.predict, pdb_csv=pdb_csv
            )

        # FOR TRAINING/VALIDATION/TESTING
        else: 
            # preliminaries for structure data
            pdb_csv = pd.read_csv(self.module_cfg.dataset.train_val_test_pdbs.csv_path)
            # apply global filters
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= self.module_cfg.dataset.train_val_test_pdbs.max_num_res]
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= self.module_cfg.dataset.train_val_test_pdbs.min_num_res]
            pdb_csv = pdb_csv[pdb_csv.Resolution <= self.module_cfg.dataset.train_val_test_pdbs.max_resolution]

            self.singles, self.ensembles = {}, {}

            # train split
            train_cfg = self.module_cfg.dataset.train
            if self.module_cfg.dataset.train_val_test_pdbs.subset is not None: # sample first n items of dataset
                train_csv = pdb_csv[pdb_csv.split == 'train'].iloc[:int(self.module_cfg.dataset.train_val_test_pdbs.subset)]
            else:
                train_csv = pdb_csv[pdb_csv.split == 'train']
  
            if train_cfg.single_struc:
                self.singles['train'] = SingleStructureDataset(cfg=train_cfg.dataset, pdb_csv=train_csv)
                self._log.info(f'{len(self.singles["train"])} TRAINING single pdb structures.')
            if train_cfg.ensemble:
                self.ensembles['train'] = EnsembleDataset(cfg=train_cfg.dataset, pdb_csv=train_csv)
                self._log.info(f'{len(self.ensembles["train"])} TRAINING pdb ensembles.')
            
            # val split
            valid_cfg = self.module_cfg.dataset.valid
            if valid_cfg.single_struc:
                self.singles['valid'] = SingleStructureDataset(cfg=valid_cfg.dataset, pdb_csv=pdb_csv[pdb_csv.split == 'val'])
                self._log.info(f'{len(self.singles["valid"])} VALIDATION single pdb structures; will sub-sample {valid_cfg.single_struc_sampler.num_struc_samples} per validation run.')
            if valid_cfg.ensemble:
                self.ensembles['valid'] = EnsembleDataset(cfg=valid_cfg.dataset, pdb_csv=pdb_csv[pdb_csv.split == 'val'])
                self._log.info(f'Will also generate {len(self.ensembles["valid"])} ensembles per validation run.')
            
            # extra val split
            extra_valid_cfg = self.module_cfg.dataset.extra_valid
            if extra_valid_cfg.use_extra:
                if valid_cfg.single_struc:
                    self.singles['extra_valid'] = SingleStructureDataset(cfg=extra_valid_cfg.dataset, pdb_csv=pdb_csv[pdb_csv.split == 'val_extra'])
                    self._log.info(f'{len(self.singles["extra_valid"])} extra VALIDATION single pdb structures; will sub-sample {valid_cfg.single_struc_sampler.num_struc_samples} per validation run.')
                if valid_cfg.ensemble:
                    self.ensembles['extra_valid'] = EnsembleDataset(cfg=extra_valid_cfg.dataset, pdb_csv=pdb_csv[pdb_csv.split == 'val_extra'])
                    self._log.info(f'Will also generate {len(self.ensembles["extra_valid"])} extra ensembles per validation run.')
            
            # test split
            test_cfg = self.module_cfg.dataset.test
            if test_cfg.single_struc:
                self.singles['test'] = SingleStructureDataset(cfg=test_cfg.dataset, pdb_csv=pdb_csv[pdb_csv.split == 'test'])
                self._log.info(f'{len(self.singles["test"])} TESTING single pdb structures; will sub-sample {test_cfg.single_struc_sampler.num_struc_samples} per test run.')
            if test_cfg.ensemble:
                self.ensembles['test'] = EnsembleDataset(cfg=test_cfg.dataset, pdb_csv=pdb_csv[pdb_csv.split == 'test'])
                self._log.info(f'Will also generate {len(self.ensembles["test"])} ensembles per test run.')

            # extra test split
            extra_test_cfg = self.module_cfg.dataset.extra_test
            if extra_test_cfg.use_extra:
                if test_cfg.single_struc:
                    self.singles['extra_test'] = SingleStructureDataset(cfg=extra_test_cfg.dataset, pdb_csv=pdb_csv[pdb_csv.split == 'test_extra'])
                    self._log.info(f'{len(self.singles["extra_test"])} extra TESTING single pdb structures; will sub-sample {test_cfg.single_struc_sampler.num_struc_samples} per test run.')
                if test_cfg.ensemble:
                    self.ensembles['extra_test'] = EnsembleDataset(cfg=extra_test_cfg.dataset, pdb_csv=pdb_csv[pdb_csv.split == 'test_extra'])
                    self._log.info(f'Will also generate {len(self.ensembles["extra_test"])} extra ensembles per test run.')

    def get_dataloader(self, stage, rank=None, num_replicas=None):
        
        num_workers = self.module_cfg.module.loaders.num_workers
        prefetch_factor = None if num_workers == 0 else self.module_cfg.module.loaders.prefetch_factor
        persistent_workers = self.module_cfg.module.loaders.persistent_workers
        
        single_struc_pdbs, ensemble_pdbs, preds = None, None, None
        if stage == 'sample':
            pred_sampler_cfg = self.module_cfg.dataset.predict.pred_sampler
            preds = self.preds
        else:
            single_struc_sampler_cfg = self.module_cfg.dataset[stage].single_struc_sampler
            ensemble_sampler_cfg = self.module_cfg.dataset[stage].ensemble_sampler
            single_struc_pdbs = self.singles.get(stage)
            ensemble_pdbs = self.ensembles.get(stage)

        single_struc_loader, ensemble_loader = None, None

        #### for debugging no distributed######
        '''
        print('HELLOOO DEBUGGING')
        use_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        if single_struc_pdbs is not None:
            if use_distributed:
                sampler = DistributedSampler(
                    single_struc_pdbs,
                    shuffle=single_struc_sampler_cfg.shuffle
                )
                shuffle = False
            else:
                sampler = None
                shuffle = single_struc_sampler_cfg.shuffle

            collate_fn = heterogenos_length_collate
            single_struc_loader = DataLoader(
                single_struc_pdbs,
                sampler=sampler,
                shuffle=shuffle if sampler is None else False,
                batch_size=single_struc_sampler_cfg.batch_size,
                drop_last=single_struc_sampler_cfg.drop_last,
                collate_fn=collate_fn,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=False,
                persistent_workers=persistent_workers
            )

        if preds is not None:
            if use_distributed:
                sampler = DistributedSampler(
                    preds,shuffle=False)
            else:
                sampler = None

            collate_fn = heterogenos_length_collate
            preds_loader = DataLoader(preds,
                                      sampler=sampler,
                                      batch_size=pred_sampler_cfg.batch_size,
                                      drop_last=False,
                                      collate_fn=collate_fn,
                                      num_workers=num_workers,
                                      prefetch_factor=prefetch_factor,
                                      pin_memory=False,
                                      persistent_workers=persistent_workers
            )
        '''
        
        if single_struc_pdbs is not None:
            if stage == 'train' and getattr(single_struc_sampler_cfg, "use_cluster_batch_sampler", False):
                batch_sampler = DistributedTrainBatchSampler(
                    dataset=single_struc_pdbs,
                    bsampler_cfg=single_struc_sampler_cfg,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=single_struc_sampler_cfg.shuffle,
                    seed=getattr(single_struc_sampler_cfg, "seed", 0),
                    drop_last=single_struc_sampler_cfg.drop_last,
                )

                single_struc_loader = DataLoader(
                    single_struc_pdbs,
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    pin_memory=False,
                    persistent_workers=persistent_workers,
                    collate_fn=heterogenos_length_collate,
                )
            else:
                sampler = DistributedSampler(single_struc_pdbs, shuffle=single_struc_sampler_cfg.shuffle)
                single_struc_loader = DataLoader(
                    single_struc_pdbs,
                    sampler=sampler,
                    batch_size=single_struc_sampler_cfg.batch_size,
                    drop_last=single_struc_sampler_cfg.drop_last,
                    collate_fn=heterogenos_length_collate,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    pin_memory=False,
                    persistent_workers=persistent_workers,
                )

        
        if ensemble_pdbs is not None:
            sampler = DistributedSampler(ensemble_pdbs, shuffle=ensemble_sampler_cfg.shuffle)
            collate_fn = heterogenos_length_collate_ensemble
            ensemble_loader = DataLoader(ensemble_pdbs,
                                         sampler=sampler,
                                         batch_size=ensemble_sampler_cfg.batch_size,
                                         num_workers=num_workers,
                                         prefetch_factor=prefetch_factor,
                                         pin_memory=False,
                                         persistent_workers=persistent_workers,
                                         collate_fn=collate_fn
            )

        if preds is not None:
            sampler = DistributedSampler(preds, shuffle=False)
            collate_fn = heterogenos_length_collate
            preds_loader = DataLoader(preds,
                                      sampler=sampler,
                                      batch_size=pred_sampler_cfg.batch_size,
                                      drop_last=False,
                                      collate_fn=collate_fn,
                                      num_workers=num_workers,
                                      prefetch_factor=prefetch_factor,
                                      pin_memory=False,
                                      persistent_workers=persistent_workers
            )
        
        if preds is not None: # predict dataset
            return preds_loader
        else:
            iterables = {
                'single_struc' : single_struc_loader, 
                'ensemble' : ensemble_loader
                }
            # REMOVE None loaders
            iterables = {k: v for k, v in iterables.items() if v is not None}
            print(f'{stage} CombinedLoader containing keys: {", ".join(iterables.keys())}')
            return CombinedLoader(iterables, 'max_size')

    def train_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('train', rank=rank, num_replicas=num_replicas)

    def val_dataloader(self, rank=None, num_replicas=None):
        if self.module_cfg.dataset.extra_valid.use_extra:
            main_loader = self.get_dataloader('valid', rank=rank, num_replicas=num_replicas)
            extra_loader = self.get_dataloader('extra_valid', rank=rank, num_replicas=num_replicas)
            return [main_loader, extra_loader]
        else:
            return self.get_dataloader('valid', rank=rank, num_replicas=num_replicas)

    def test_dataloader(self, rank=None, num_replicas=None):
        if self.module_cfg.dataset.extra_test.use_extra:
            main_loader = self.get_dataloader('test', rank=rank, num_replicas=num_replicas)
            extra_loader = self.get_dataloader('extra_test', rank=rank, num_replicas=num_replicas)
            return [main_loader, extra_loader]
        else:
            return self.get_dataloader('test', rank=rank, num_replicas=num_replicas)

    def predict_dataloader(self, rank=None, num_replicas=None):
        return self.get_dataloader('sample', rank=rank, num_replicas=num_replicas)
