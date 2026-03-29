"""
precompute_stats.py
-------------------
Serialises:
  - shared_pca.pkl      : shared PCA model
  - dataset_stats.pkl   : per-dataset Gaussian stats (mu, var)

Run once:
    python precompute_stats.py
Then run build_embeddings.py once to generate embeddings.pkl.
"""

import os, pickle
import numpy as np
from sklearn.decomposition import PCA

SAVED_FEATURES_PATH = '/shared/ssd_28T/home/mm3572/anomaly_detection/Mousa/SavedFeatures'
OUTPUT_DIR          = '/shared/ssd_28T/home/mm3572/anomaly_detection/Mousa/web_app/static/precomputed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXCLUDE_DATASETS   = {'STARAE', 'dataset_STARAE'}
KL_VARIANCE_TARGET = 0.95

def dims_for_variance(pca, threshold=0.95):
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumvar >= threshold) + 1)

# ── Load features ─────────────────────────────────────────────────────────
print('===> Loading features ...')
all_feats, all_names, seen = [], [], set()

for fname in sorted(os.listdir(SAVED_FEATURES_PATH)):
    if not (fname.endswith('.pkl') or fname.endswith('.pickle')):
        continue
    if not fname.startswith('SavedFeatures_Retfound_'):
        continue
    ds_name = (fname.replace('SavedFeatures_Retfound_', '')
                    .replace('.pkl', '').replace('.pickle', ''))
    if ds_name in EXCLUDE_DATASETS:
        print(f'  EXCLUDED : {ds_name}'); continue
    if ds_name in seen:
        print(f'  DUPLICATE: {ds_name}'); continue
    with open(os.path.join(SAVED_FEATURES_PATH, fname), 'rb') as f:
        d = pickle.load(f)
    feats = d['Features']
    if feats.shape[0] < 20:
        print(f'  TOO FEW  : {ds_name}'); continue
    all_feats.append(feats)
    all_names.append(ds_name)
    seen.add(ds_name)
    print(f'  Loaded   : {ds_name:<40s}  {feats.shape[0]:5d} images')

X_all = np.vstack(all_feats)
print(f'\n===> Total samples : {X_all.shape[0]}  |  Datasets: {len(all_names)}')

# ── KL PCA dims ───────────────────────────────────────────────────────────
KL_PCA_DIMS, worst_ds = 0, ''
for name, feats in zip(all_names, all_feats):
    pca_tmp  = PCA().fit(feats)
    d_needed = dims_for_variance(pca_tmp, KL_VARIANCE_TARGET)
    if d_needed > KL_PCA_DIMS:
        KL_PCA_DIMS, worst_ds = d_needed, name
KL_PCA_DIMS = max(20, min(KL_PCA_DIMS, 200))
print(f'===> KL_PCA_DIMS = {KL_PCA_DIMS}  (worst: {worst_ds})')

# ── Fit shared PCA ────────────────────────────────────────────────────────
print(f'===> Fitting shared PCA ({KL_PCA_DIMS}D) ...')
pca_shared = PCA(n_components=KL_PCA_DIMS, random_state=42).fit(X_all)
print(f'===> Shared PCA retains {pca_shared.explained_variance_ratio_.sum()*100:.1f}% variance')

# ── Per-dataset Gaussian stats ────────────────────────────────────────────
print('===> Computing per-dataset Gaussian statistics ...')
dataset_stats, dataset_sizes = {}, {}
offset = 0
for name, feats in zip(all_names, all_feats):
    n     = feats.shape[0]
    X_pca = pca_shared.transform(feats)
    dataset_stats[name] = (X_pca.mean(axis=0), X_pca.var(axis=0))
    dataset_sizes[name] = n
    offset += n

# ── Save ──────────────────────────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, 'shared_pca.pkl'), 'wb') as f:
    pickle.dump(pca_shared, f, protocol=4)
with open(os.path.join(OUTPUT_DIR, 'dataset_stats.pkl'), 'wb') as f:
    pickle.dump({
        'dataset_stats' : dataset_stats,
        'dataset_sizes' : dataset_sizes,
        'kl_pca_dims'   : KL_PCA_DIMS,
        'var_target'    : KL_VARIANCE_TARGET,
        'dataset_names' : all_names,
    }, f, protocol=4)

print(f'\n===> Done. Now run: python build_embeddings.py')
