'''Explore the CaeNDR app RData chunks (Bell & Paaby 2024): print object names, shapes, columns,
and a head, so we can locate the per-strain expression matrix and extract the 14 channel genes.
Run on  (pyreadr installed): python3 _b5_explore_rdata.py'''
import pyreadr, sys, glob
WB14 = ['WBGene00001187','WBGene00006742','WBGene00000367','WBGene00001171','WBGene00001202',
        'WBGene00002235','WBGene00002149','WBGene00003558','WBGene00014261','WBGene00022240',
        'WBGene00002242','WBGene00007176','WBGene00004830','WBGene00004831']  # the 14 channel genes
for f in sorted(glob.glob('/root/autodl-tmp/cendr_chunk*.RData')):
    print('='*60, f)
    try:
        d = pyreadr.read_r(f)
    except Exception as e:
        print('  read error:', repr(e)[:120]); continue
    for name, obj in d.items():
        print(f'  OBJECT {name!r}: shape={getattr(obj,"shape",None)}')
        try:
            print('   columns:', list(obj.columns)[:20])
            print('   head:\n', obj.head(2).to_string()[:800])
            # search for any of the 14 WBGene IDs anywhere in the frame
            hits = 0
            for col in obj.columns:
                try:
                    if obj[col].astype(str).isin(WB14).any(): hits += 1; print(f'   *** col {col!r} contains WBGene IDs')
                except Exception: pass
            if hits == 0: print('   (no WBGene IDs found in columns)')
        except Exception as e:
            print('   (not a frame / inspect err)', repr(e)[:80])
