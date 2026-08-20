"""
Script to pre-filter antibody data. Run on many CPU cores.
Sets up files to be used in pre-processing PDBs (process_ab_pdb_files.py).

* use ImmuneBuilder dataset from https://zenodo.org/records/7258553
* merge with pre-defined train/val/test splits from https://zenodo.org/records/8164693
* filter out instances with inappropriate loci, intrusive stop codons, frameshifts
* reconstruct full sequences from ANARCI numbering
* keep instances where reconstructed sequence matches full sequence in entry
* keep instances with sequence length between MIN_SEQ_LEN and MAX_SEQ_LEN
* eliminate sequences with missing PDBs

Usage:
python pre_filter_ab_data.py
"""
import pandas as pd
import numpy as np
import os

from multiprocessing import Pool
#from utils import ANARCIs_to_sequence


NUM_PROC = 8 
MIN_SEQ_LEN = 200
MAX_SEQ_LEN = 270


#all_metadata_csv = "/vols/opig/users/vavourakis/data/OAS_models/OAS_paired_all.csv" 
#splits_csv = "/vols/opig/users/vavourakis/data/oas_splits.csv"
#strucs_dir = "/vols/opig/users/vavourakis/data/OAS_models/structures"
#out_filtered_csv = "/vols/opig/users/vavourakis/data/OAS_models/OAS_paired_prefiltered.csv"

all_metadata_csv = "/opig-shared/users/lina4783/data/split_v01/df_filtered_ab_split_v4.csv"
strucs_dir = "/opig-shared/users/lina4783/data/split_v01"
out_filtered_csv="/opig-shared/users/lina4783/data/prefiltered_dataset_FINAL2.csv"

#def process_row(x):
    #return ANARCIs_to_sequence(x.VH_numbering_list, x.VL_numbering_list)

def split_chain_into_regions(seq, cdrs):#given full chain and CDRs, split into framework and cdrs
    if len(seq)==0:
        return ["","","","","","",""]
    regions = []
    start = 0
    for cdr in cdrs:
        if not cdr:
            print(seq)
            print(cdrs)
            raise ValueError("Found empty CDR for a non-missing chain")
        
        i = seq.find(cdr, start)
        if i == -1:
            raise ValueError(f"Could not find CDR {cdr} in sequence")
        regions.append(seq[start:i])#framework before this CDR
        regions.append(cdr)#the CDR itself
        start = i + len(cdr)#new start
    regions.append(seq[start:])#final framework
    if len(regions) != 7:
        raise ValueError(f"Expected 7 regions, got {len(regions)}")
    return regions


def process_row(x):
    seqs_h = split_chain_into_regions(x.VH_numerable_seq, [x.CDRH1, x.CDRH2, x.CDRH3])
    seqs_l = split_chain_into_regions(x.VL_numerable_seq, [x.CDRL1, x.CDRL2, x.CDRL3])

    idx_h = list(__import__("numpy").cumsum([len(s) for s in seqs_h]))
    idx_l = [1000 + i for i in __import__("numpy").cumsum([len(s) for s in seqs_l])]

    cdr_concat = seqs_h[1] + seqs_h[3] + seqs_h[5] + seqs_l[1] + seqs_l[3] + seqs_l[5]
    fw_concat = seqs_h[0] + seqs_h[2] + seqs_h[4] + seqs_l[0] + seqs_l[2] + seqs_l[4]
    full_seq = "/".join(["".join(seqs_h), "".join(seqs_l)])
    seqlen = len("".join(seqs_h)) + len("".join(seqs_l))

    return (
        *seqs_h, *seqs_l,
        full_seq,
        cdr_concat,
        fw_concat,
        (0, *idx_h, 1000, *idx_l),#indexing to match original function
        seqlen,
    )

def add_validation(df: pd.DataFrame,cluster_col: str = "ab_clusters",base_split_col: str = "ab_split",
    new_split_col: str = "split",val_frac_of_train_clusters: float = 0.125,#overall 10% validation
    random_state: int = 0) -> pd.DataFrame:
    '''creates new split column which inclusdes validation set (taken from training)'''
    out = df.copy()
    #print(len(out))

    #existing split labels
    out[new_split_col] = out[base_split_col].astype(str)

    #only take validation from clusters in trianing
    train_mask = out[base_split_col].eq("train")
    train_clusters = (out.loc[train_mask, cluster_col].dropna().astype(str).unique())

    rng = np.random.default_rng(random_state)
    n_val_clusters = max(1, int(round(len(train_clusters) * val_frac_of_train_clusters)))#assumes clusters of roughly equal size
    val_clusters = set(rng.choice(train_clusters, size=n_val_clusters, replace=False))

    #reassign those entire clusters to validation
    val_mask = train_mask & out[cluster_col].astype(str).isin(val_clusters)
    out.loc[val_mask, new_split_col] = "val"
    #print(len(out))
    return out

if __name__ == '__main__':

    print('\nreading metadata')
    #load metadata csv
    df = pd.read_csv(all_metadata_csv)
    #renames column of full sequence


    #######removed merging with data splits df bc splits already in df######

    print(f'\n{len(df)} entries before filtering')
    print('filtering nan H and L sequences')
    #df = df[df['VH_numerable_seq'].notna() & df['VL_numerable_seq'].notna() &
        #df['CDRH1'].notna() & df['CDRH2'].notna() & df['CDRH3'].notna() &
        #df['CDRL1'].notna() & df['CDRL2'].notna() & df['CDRL3'].notna()]
    df = df[~(df['VH_numerable_seq'].isna() & df['VL_numerable_seq'].isna())]#remove cols where both chains are missing
    to_fill=['VH_numerable_seq','VL_numerable_seq','CDRH1','CDRH2','CDRH3','CDRL1','CDRL2','CDRL3']
    df[to_fill]=df[to_fill].fillna('')#fill nans with empty string


    #df.rename(columns={'full_seq': 'full_seq_orig'}, inplace=True)
    #add full sequence column
    df['full_seq_orig'] = df['VH_numerable_seq'] + '/' + df['VL_numerable_seq']
    df['seqlen_orig'] = df['full_seq_orig'].str.len()

    #df.to_csv('/opig-shared/users/lina4783/data/pre_len_remove.csv')


    print(f'{len(df)} entries after filtering\n')

    print(f'keeping sequences with length between {MIN_SEQ_LEN} and {MAX_SEQ_LEN} or 80 and 140.')
    df = df[((df['seqlen_orig'] >= MIN_SEQ_LEN) & (df['seqlen_orig'] <= MAX_SEQ_LEN)) | ((df['seqlen_orig'] >= 80) & (df['seqlen_orig'] <= 140))]  

    #remove extra long cdrh3s
    print('removing sequences with CDRH3 >30 length\n')
    df=df[df.CDRH3.str.len()<=30] 

    print(f'\n{len(df)} entries remaining\n')
    
    print(f'wrangling sequences')
    with Pool(processes=NUM_PROC) as pool:
        results = pool.map(process_row, [row for _, row in df.iterrows()])
    df2 = pd.DataFrame(results, columns=['fwr1_h', 'cdr1_h', 'fwr2_h', 'cdr2_h', 'fwr3_h', 'cdr3_h', 'fwr4_h',
                                         'fwr1_l', 'cdr1_l', 'fwr2_l', 'cdr2_l', 'fwr3_l', 'cdr3_l', 'fwr4_l',
                                         'full_seq', 'concat_CDR', 'fw_concat', 'region_indices', 'seqlen']
                       )
    #print(df2.iloc[0])

    print(f'merging')
    df2.index = df.index
    df = pd.concat([df, df2], axis=1)
    del df2
  
    #does more filtering
    print(f'keeing instances where reconstructed sequence matches original full sequence')
    df = df[df['full_seq'] == df['full_seq_orig']]
    print(f'{len(df)} entries remaining\n')
 

    #adds column pointing to location of file
    print(f'generating and checking structure paths')
    df['structure_path'] = df['INSTANCE'].apply(lambda x: os.path.join(strucs_dir, f'{x}.cif'))
    file_mask = df['structure_path'].apply(lambda path: os.path.exists(path))
    df = df[file_mask]

    print(f'\n{len(df)} entries remaining\n')

    cols_to_keep = ['INSTANCE','PDB_ID', 'SABDAB_ID','structure_path', 'seqlen', 'resolution','cdrh3_cluster',#TODO: verify which are needed
       'cdrh123_cluster', 'cdrl123_cluster', 'ab_cluster', 'ab_split', 'full_seq','holo', 'Hchain', 'Lchain', 'VH_numerable_seq','VL_numerable_seq',
                    'fwr1_h', 'cdr1_h', 'fwr2_h', 'cdr2_h', 'fwr3_h', 'cdr3_h', 'fwr4_h',
                    'fwr1_l', 'cdr1_l', 'fwr2_l', 'cdr2_l', 'fwr3_l', 'cdr3_l', 'fwr4_l',
                    'concat_CDR', 'fw_concat', 'region_indices'] 
                    #,'ANARCI_numbering_heavy', 'ANARCI_numbering_light']
    df = df[cols_to_keep]

    #create new split column which includes a validation set
    print('adding validation set')
    df = add_validation(df,cluster_col="ab_cluster",base_split_col="ab_split",new_split_col="split",
                                 val_frac_of_train_clusters=0.15)

    total=len(df)
    print(f'{total} datapoints')
    print('% validation: ',len(df[df.split=='val'])/total)
    print('% train: ',len(df[df.split=='train'])/total)
    print('% test: ',len(df[df.split=='test'])/total)
    
    #with_val_df.to_csv('/opig-shared/users/lina4783/data/with_val_debug.csv')

    #rename columns 
    print('renaming columns...')
    col_name_map={'resolution':'Resolution','ab_cluster':'cluster_ids','cdr1_h':'CDRH1','cdr2_h':'CDRH2',
                  'cdr3_h':'CDRH3','cdr1_l':'CDRL1','cdr2_l':'CDRL2','cdr3_l':'CDRL3',
                  'VH_numerable_seq':'VH_seq','VL_numerable_seq':'VL_seq','PDB_ID':'ID'}

    df=df.rename(columns=col_name_map)#TODO: uncomment

    print(f'writing to {out_filtered_csv}')
    df.to_csv(out_filtered_csv, index=False)
    print('done')
