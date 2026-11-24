import glob, pandas as pd
ms = pd.read_csv('data/processed/modality_status.csv')
print(ms.mri_status.value_counts())
for s, g in ms.groupby('mri_status'):
    print('\n', s); print(g.mri_reason.head(2).to_string())
ms['image_id'] = ms.mri_reason.astype(str).str.extract(r'image_id=(\d+)')
sel = ms[ms.image_id.notna()][['RID', 'image_id']]
dl = pd.read_csv('data/processed/mri_download_list.csv')[['RID', 'image_id']]
dl['image_id'] = dl.image_id.astype(str).str.replace(r'\D', '', regex=True)
allsel = pd.concat([sel, dl]).drop_duplicates('RID')
print('\nIDs from modality_status:', len(sel), '| from download list:', len(dl), '| union:', len(allsel))
for p in glob.glob('**/selected_MRI_records_90d.csv', recursive=True):
    print(p, pd.read_csv(p, nrows=2).columns.tolist())
allsel.to_csv('data/processed/mri_selected.csv', index=False)