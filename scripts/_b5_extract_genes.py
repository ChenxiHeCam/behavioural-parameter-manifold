'''Extract the 14 channel genes x strain expression (meanexp, vst) from the CaeNDR RData chunks
(Bell & Paaby 2024, long format). Concatenate across whatever chunks are present, pivot to a
strain x gene matrix, save a small CSV. Run on : python3 _b5_extract_genes.py'''
import pyreadr, glob, sys
import pandas as pd
WB14 = ['WBGene00001187','WBGene00006742','WBGene00000367','WBGene00001171','WBGene00001202',
        'WBGene00002235','WBGene00002149','WBGene00003558','WBGene00014261','WBGene00022240',
        'WBGene00002242','WBGene00007176','WBGene00004830','WBGene00004831']
parts = []
for f in sorted(glob.glob('/root/autodl-tmp/cendr_chunk*.RData')):
    try:
        d = pyreadr.read_r(f)
    except Exception as e:
        print(f'SKIP {f}: {repr(e)[:60]}'); continue
    df = list(d.values())[0]
    sub = df[df['gene_id'].isin(WB14)][['strain', 'gene_id', 'meanexp']].drop_duplicates(['strain', 'gene_id'])
    parts.append(sub); print(f'{f}: {len(sub)} (strain,gene) rows, {sub["strain"].nunique()} strains')
allsub = pd.concat(parts).drop_duplicates(['strain', 'gene_id'])
mat = allsub.pivot(index='strain', columns='gene_id', values='meanexp')
mat.to_csv('/root/autodl-tmp/cendr_14genes_by_strain.csv')
print(f'\nMATRIX: {mat.shape[0]} strains x {mat.shape[1]} genes')
print('genes present:', list(mat.columns))
print('missing of 14:', [g for g in WB14 if g not in mat.columns])
print('SAVED /root/autodl-tmp/cendr_14genes_by_strain.csv')
