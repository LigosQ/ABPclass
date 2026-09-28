#!/usr/bin/env python3
"""
main.py - end-to-end pipeline: in-fold MRMD3.0 feature selection + Transformer/RBF-SVM ensemble
=====================================================================================================
Input features = ESM2 + ProtT5-XL + ACC + CC + nmbroto
The whole pipeline sits inside an outer 5-fold CV: each fold performs feature selection
and training on the training fold (80%) only, then evaluates the held-out fold (20%).
Fold results are aggregated into ACC/MCC/AUC, plus a pooled AUC over the concatenated folds.

Feature selection inside a fold (supervised but leak-free; uses only the fold's training labels):
  1) Pre-filter: |Pearson corr with class| (training-fold y), keep top-PREFILTER
  2) MRMD ranking: the official run() computes Euclidean/Cosine/Tanimoto/Person distances,
           giving the MRMD-Eu/Cos/Tan rankings; keep the top-1020 single features
           NOTE: the Person corr with class inside run() is an intrinsic part of MRMD
  3) Second-level crosses: pairwise crosses of the MRMD top-200 single features (19900),
           pre-filtered to top-300 by |Pearson corr|, then ranked by run() and cut to top-30
  4) Training: two-stage Transformer (AdamW + cross-entropy, then higher dropout + focal
           loss) plus an RBF-SVM; the two probabilities are averaged before evaluation

Usage:
  python3 main.py --test --prefilter 250      # quick test (2 folds; the shipped sample suffices)
  python3 main.py --prefilter 600             # full run (5 folds; needs the complete dataset)
  python3 main.py --variant euc|cos|tan       # choose the MRMD variant
"""
import sys, os, math, random, time, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Data directory: defaults to <repo>/data (the repo ships a 5% sampled example dataset);
# set the DATA_DIR environment variable to the directory holding the complete feature CSVs
# to run at full scale (it must mirror the same subdirectory structure).
BASE = os.environ.get('DATA_DIR', os.path.join(SCRIPT_DIR, 'data'))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'mrmd3_official'))
os.environ['JOBLIB_CPU_COUNT'] = '4'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef, f1_score, roc_auc_score
from sklearn.svm import SVC
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

# On macOS the default 'spawn' start method makes multiprocessing.Pool children re-run the
# top-level code of this script (infinite recursion). The official MRMD run() uses a Pool
# internally, so 'fork' is needed: children inherit memory and skip the top-level code.
try:
    mp.set_start_method('fork', force=True)
except RuntimeError:
    pass

test_mode = '--test' in sys.argv
VARS = [a for a in sys.argv if a.startswith('--variant')]
if VARS:
    if '=' in VARS[0]:
        VARIANT = VARS[0].split('=')[1]
    else:
        VARIANT = sys.argv[sys.argv.index(VARS[0]) + 1]
else:
    VARIANT = 'euc'
PF = [a for a in sys.argv if a.startswith('--prefilter')]
if PF:
    if '=' in PF[0]:
        PREFILTER = int(PF[0].split('=')[1])
    else:
        PREFILTER = int(sys.argv[sys.argv.index(PF[0]) + 1])
else:
    PREFILTER = 250 if test_mode else 600

SEED_BASE = 42; DL_SEED = 10059
if test_mode:
    EA, EB, N_FOLDS = 5, 3, 2
else:
    EA, EB, N_FOLDS = 40, 25, 5
K_USE = 1020; N_USE = 30; TOP200 = 200

def load_source(pa, pb, sl=None):
    da = pd.read_csv(pa); db = pd.read_csv(pb)
    if db['class'].dtype == float: db['class'] = db['class'].astype(int)
    fa = [c for c in da.columns if c != 'class']; fb = [c for c in db.columns if c != 'class']
    if sl is not None: fa = fa[sl]; fb = fb[sl]
    Xa = np.nan_to_num(da[fa].values.astype(np.float64), 0)
    Xb = np.nan_to_num(db[fb].values.astype(np.float64), 0)
    ma = Xa.mean(1, keepdims=1); sa = Xa.std(1, keepdims=1) + 1e-8
    mb = Xb.mean(1, keepdims=1); sb = Xb.std(1, keepdims=1) + 1e-8
    return ((Xa - ma) / sa).astype(np.float32), ((Xb - mb) / sb).astype(np.float32), \
           da['class'].values.astype(np.int64), db['class'].values.astype(np.int64)

def load_esm2_only(pa, pb):
    da = pd.read_csv(pa); db = pd.read_csv(pb)
    if db['class'].dtype == float: db['class'] = db['class'].astype(int)
    fa = [c for c in da.columns if c.startswith('esm_f_')]
    fb = [c for c in db.columns if c.startswith('esm_f_')]
    Xa = np.nan_to_num(da[fa].values.astype(np.float64), 0)
    Xb = np.nan_to_num(db[fb].values.astype(np.float64), 0)
    ma = Xa.mean(1, keepdims=1); sa = Xa.std(1, keepdims=1) + 1e-8
    mb = Xb.mean(1, keepdims=1); sb = Xb.std(1, keepdims=1) + 1e-8
    return ((Xa - ma) / sa).astype(np.float32), ((Xb - mb) / sb).astype(np.float32), \
           da['class'].values.astype(np.int64), db['class'].values.astype(np.int64)

def load_physchem_subgroup(pa, pb, keep_fn):
    da = pd.read_csv(pa); db = pd.read_csv(pb)
    if db['class'].dtype == float: db['class'] = db['class'].astype(int)
    fa = [c for c in da.columns if keep_fn(c)]
    fb = [c for c in db.columns if keep_fn(c)]
    Xa = np.nan_to_num(da[fa].values.astype(np.float64), 0)
    Xb = np.nan_to_num(db[fb].values.astype(np.float64), 0)
    ma = Xa.mean(1, keepdims=1); sa = Xa.std(1, keepdims=1) + 1e-8
    mb = Xb.mean(1, keepdims=1); sb = Xb.std(1, keepdims=1) + 1e-8
    return ((Xa - ma) / sa).astype(np.float32), ((Xb - mb) / sb).astype(np.float32)

print("\nLoading features...")
xa_t5xl, xb_t5xl, _, _ = load_source(os.path.join(BASE, 'protT5xl/t5onnx_amp_2merged.csv'),
                                     os.path.join(BASE, 'protT5xl/t5onnx_biof_2merged.csv'))
xa_esm, xb_esm, ya, yb = load_esm2_only(os.path.join(BASE, '2Feats/Amp_5props_pogneg.csv'),
                                        os.path.join(BASE, '2Feats/biofilm_posneg_comb.csv'))
AMP_2F = os.path.join(BASE, '2Feats/Amp_5props_pogneg.csv')
BIO_2F = os.path.join(BASE, '2Feats/biofilm_posneg_comb.csv')
xa_acc, xb_acc = load_physchem_subgroup(AMP_2F, BIO_2F, lambda c: c.startswith('ACC_prop_'))
xa_cc, xb_cc = load_physchem_subgroup(AMP_2F, BIO_2F, lambda c: c.startswith('AC_prop_'))
xa_nmb, xb_nmb = load_physchem_subgroup(AMP_2F, BIO_2F, lambda c: '.lag' in c)

FEAT_POOL = {'ESM2': (xa_esm, xb_esm), 'ProtT5-XL': (xa_t5xl, xb_t5xl),
             'ACC': (xa_acc, xb_acc), 'CC': (xa_cc, xb_cc), 'nmbroto': (xa_nmb, xb_nmb)}
HSTACK_ORDER = ['ESM2', 'ProtT5-XL', 'ACC', 'CC', 'nmbroto']

Xa_parts, Xb_parts, dims = [], [], {}
for grp in HSTACK_ORDER:
    xa, xb = FEAT_POOL[grp]
    Xa_parts.append(xa); Xb_parts.append(xb)
    dims[grp] = xa.shape[1]
Xa_all = np.hstack(Xa_parts).astype(np.float32)
Xb_all = np.hstack(Xb_parts).astype(np.float32)
print(f"Raw dims: {dims} -> total={Xa_all.shape[1]}")

# ---- official MRMD3.0 integration ----
from feature_selection.MRMD import run as mrmd_run, run_s as mrmd_run_s
class _Logger:
    def info(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def warning(self, *a, **k): pass
_LOGGER = _Logger()

def _write_csv_for_mrmd(X, y, cols_idx, path):
    sub = X[:, cols_idx]
    df = pd.DataFrame(sub, columns=[f'f{i}' for i in range(len(cols_idx))])
    df.insert(0, 'class', y)
    df.to_csv(path, index=False)

def fs_mrmd3(Xb, yb, variant, prefilter):
    """Official MRMD3.0 feature selection: supervised pre-filter (Pearson corr with class,
    using the passed yb) -> official run() for the Euc/Cos/Tan rankings -> top-K_USE single
    features, then top200 crosses cut to top-N_USE.
    NOTE: when called inside a fold, yb holds the training-fold labels (80%), so the
    pre-filter is supervised but leak-free. run() also computes Person corr with class
    internally, which is part of the MRMD relevance term."""
    const_mask = np.std(Xb, axis=0) < 1e-12
    cand = np.where(~const_mask)[0]
    # supervised pre-filter: top-prefilter by |Pearson corr with class| (training-fold y)
    pcorr = np.abs(np.corrcoef(Xb[:, cand].T, yb)[:-1, -1])
    pcorr = np.nan_to_num(pcorr, 0)
    top_pre = cand[np.argsort(pcorr)[::-1][:prefilter]]
    print(f"  [prefilter] {len(top_pre)} candidates kept (SUPERVISED Pearson corr w/ fold-y)")
    tmp = os.path.join(SCRIPT_DIR, f'_mrmd3_tmp_{os.getpid()}.csv')
    _write_csv_for_mrmd(Xb, yb, top_pre, tmp)
    t0 = time.time()
    mrmd_c, mrmd_e, mrmd_t = mrmd_run(tmp, _LOGGER)   # official full run, three rankings
    print(f"  [MRMD3.0] 3-rank done in {time.time()-t0:.0f}s")
    os.remove(tmp) if os.path.exists(tmp) else None
    vmap = {'euc': mrmd_e, 'cos': mrmd_c, 'tan': mrmd_t}
    ranked = vmap[variant]
    name2idx = {f'f{i}': top_pre[i] for i in range(len(top_pre))}
    ranked_idx = [name2idx[nm] for nm in ranked if nm in name2idx]
    if len(ranked_idx) < K_USE:
        ranked_idx = list(ranked_idx) + [i for i in cand if i not in ranked_idx][:K_USE - len(ranked_idx)]
    ib = np.array(ranked_idx[:K_USE])
    top200 = ranked_idx[:TOP200]
    n_top = len(top200)          # cap at the number actually available when fewer than TOP200
    Xt = Xb[:, top200]
    pairs = [(i, j) for i in range(n_top) for j in range(i + 1, n_top)]
    NCT = len(pairs)
    Xc = np.zeros((len(Xb), NCT), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        Xc[:, k] = Xt[:, i] * Xt[:, j]
    # supervised pre-filter: cross features top-300 by |Pearson corr with class| (training-fold y)
    cc = np.abs(np.corrcoef(Xc.T, yb)[:-1, -1])
    cc = np.nan_to_num(cc, 0)
    top_c = np.argsort(cc)[::-1][:300]
    # official MRMD ranking (by variant), cut to top-N_USE
    tmp2 = os.path.join(SCRIPT_DIR, f'_mrmd3_tmp2_{os.getpid()}.csv')
    _write_csv_for_mrmd(Xc, yb, top_c, tmp2)
    mrmd_c2, mrmd_e2, mrmd_t2 = mrmd_run(tmp2, _LOGGER)
    os.remove(tmp2) if os.path.exists(tmp2) else None
    vmap2 = {'euc': mrmd_e2, 'cos': mrmd_c2, 'tan': mrmd_t2}
    name2idx2 = {f'f{i}': top_c[i] for i in range(len(top_c))}
    ranked_c = [name2idx2[nm] for nm in vmap2[variant] if nm in name2idx2][:N_USE]
    pairs_glob = [(int(top200[pairs[k][0]]), int(top200[pairs[k][1]])) for k in ranked_c]
    return ib, pairs_glob

def build_f(X, ib, cross_pairs):
    cols = [X[:, ib].astype(np.float32)]
    for a, b in cross_pairs:
        cols.append((X[:, [a]] * X[:, [b]]).astype(np.float32))
    return np.hstack(cols)

# ---- model definition (Transformer + positional encoding + focal loss) ----
class PE(nn.Module):
    def __init__(s, dm, ml=2048, dr=0.2):
        super().__init__(); s.dr = nn.Dropout(dr)
        pe = torch.zeros(ml, dm); pos = torch.arange(0, ml).float().unsqueeze(1)
        dv = torch.exp(torch.arange(0, dm, 2).float() * (-math.log(10000.0) / dm))
        pe[:, 0::2] = torch.sin(pos * dv); pe[:, 1::2] = torch.cos(pos * dv)
        s.register_buffer('pe', pe.unsqueeze(0))
    def forward(s, x): return s.dr(x + s.pe[:, :x.size(1), :])

class TF(nn.Module):
    def __init__(s, input_dim):
        super().__init__()
        s.ip = nn.Linear(input_dim, 128); s.idr = nn.Dropout(0.2); s.pe = PE(128, dr=0.2)
        ly = nn.TransformerEncoderLayer(128, 4, 256, 0.2, 'gelu', batch_first=True)
        s.tf = nn.TransformerEncoder(ly, 2); s.nm = nn.LayerNorm(128); s.cdr = nn.Dropout(0.2)
        s.cls = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def forward(s, x):
        x = s.ip(x); x = s.idr(x); x = x.unsqueeze(1); x = s.pe(x); x = s.tf(x)
        return s.cls(s.cdr(s.nm(x).squeeze(1)))
    def predict_proba(s, x): return F.softmax(s.forward(x), dim=1)
    def sd(s, p):
        for m in s.modules():
            if isinstance(m, nn.Dropout): m.p = p

class FL2(nn.Module):
    def __init__(s, g=3.0): super().__init__(); s.g = g
    def forward(s, i, t):
        ce = F.cross_entropy(i, t, reduction='none')
        return ((1 - torch.exp(-ce)) ** s.g * ce).mean()

DEV = torch.device('cpu')


# ---- in-fold feature selection + training (leak-free) ----
def run_training_insidecv(Xa_all, Xb_all, ya, yb, label):
    """
    Outer 5-fold CV:
      per fold: feature selection on the training fold (80%) only
                (pre-filter + official MRMD ranking + cross features)
             -> build the training / validation / AMP-augmented matrices from those features
             -> train Transformer + SVM, test on the held-out fold (20%)
    Fold confusion matrices are pooled -> ACC/F1/Sn/Sp/MCC
    """
    print(f"\n{'='*60}\n[{label}] INSIDE-CV feature selection + training\n{'='*60}")
    t0_total = time.time()
    # AMP-augmented samples (80% random undersampling on the AMP side; independent of the
    # B fold, hence leak-free)
    random.seed(SEED_BASE); np.random.seed(SEED_BASE); torch.manual_seed(SEED_BASE)
    all_idx = np.arange(len(Xa_all))
    train_idx, _ = train_test_split(all_idx, test_size=0.2, random_state=SEED_BASE, stratify=ya)
    Xat = Xa_all[train_idx]; yat = ya[train_idx]
    rus = RandomUnderSampler(sampling_strategy={0: int(np.bincount(yat)[1] * 2),
                                                 1: np.bincount(yat)[1]}, random_state=SEED_BASE)
    _, yad = rus.fit_resample(Xat, yat)
    ad_rows = rus.sample_indices_; yad_arr = yat[ad_rows]
    amp_idx_for_train = train_idx[ad_rows]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_BASE)
    splits = list(skf.split(Xb_all, yb))
    agg_cm = np.zeros((2, 2), dtype=int)
    fold_fds = []
    fold_scores, fold_labels = [], []   # collected for ROC / pooled AUC
    fold_aucs, fold_metrics = [], []    # per-fold AUC, and [ACC,F1,Sn,Sp,MCC,AUC]
    for fi, (tr, va) in enumerate(splits):
        Xbt = Xb_all[tr]; ybt = yb[tr]          # training fold (80%)
        Xbv_full = Xb_all[va]; ybv = yb[va]     # held-out fold (20%)
        # per-fold feature selection (training fold only, leak-free)
        ib, pairs_glob = fs_mrmd3(Xbt, ybt, VARIANT, PREFILTER)
        Xb_f_tr = build_f(Xbt, ib, pairs_glob)
        Xb_f_va = build_f(Xbv_full, ib, pairs_glob)
        Xa_f_ad = build_f(Xa_all[amp_idx_for_train], ib, pairs_glob)
        fd = Xb_f_tr.shape[1]; fold_fds.append(fd)
        # training (two-stage Transformer + RBF-SVM)
        n0 = np.bincount(ybt)[0]
        sm3 = BorderlineSMOTE(sampling_strategy={0: n0, 1: int(n0 / 0.33)}, random_state=SEED_BASE + fi)
        Xbu, ybu = sm3.fit_resample(Xb_f_tr, ybt)
        Xbu = Xbu.astype(np.float32); ybu = ybu.astype(np.int64)
        Xab = np.vstack([Xa_f_ad, Xbu]); yab = np.concatenate([yad_arr, ybu])
        gen = torch.Generator(); gen.manual_seed(DL_SEED)
        gen2 = torch.Generator(); gen2.manual_seed(DL_SEED)
        jl = DataLoader(list(zip(torch.tensor(Xab), torch.tensor(yab))),
                        batch_size=64, shuffle=True, generator=gen)
        bl = DataLoader(list(zip(torch.tensor(Xbu), torch.tensor(ybu))),
                        batch_size=64, shuffle=True, generator=gen2)
        m = TF(fd).to(DEV)
        opt = optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
        sc = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EA)
        for e in range(1, EA + 1):
            m.train()
            running = 0.0; nb = 0
            for Xl, yl in jl:
                Xl = Xl.to(DEV); yl = yl.to(DEV)
                opt.zero_grad(); l = F.cross_entropy(m(Xl), yl)
                l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                running += l.item(); nb += 1
            sc.step()
            if e == 1 or e % 10 == 0 or e == EA:
                print(f"    Fold{fi+1} A e{e}/{EA} loss={running/max(nb,1):.4f}")
        m.sd(0.3); fl = FL2()
        opt2 = optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4)
        sc2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EB)
        for e in range(1, EB + 1):
            m.train()
            running = 0.0; nb = 0
            for Xl, yl in bl:
                Xl = Xl.to(DEV); yl = yl.to(DEV)
                opt2.zero_grad(); l = fl(m(Xl), yl)
                l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt2.step()
                running += l.item(); nb += 1
            sc2.step()
            if e == 1 or e % 10 == 0 or e == EB:
                print(f"    Fold{fi+1} B e{e}/{EB} loss={running/max(nb,1):.4f}")
        svm = SVC(kernel='rbf', class_weight='balanced', probability=True,
                  random_state=SEED_BASE + fi); svm.fit(Xbu, ybu)
        m.eval()
        with torch.no_grad():
            ptf = m.predict_proba(torch.tensor(Xb_f_va).to(DEV)).cpu().numpy()[:, 1]
        psv = svm.predict_proba(Xb_f_va)[:, 1]; pen = (ptf + psv) / 2.0
        vp = (pen > 0.5).astype(int); cm = confusion_matrix(ybv, vp); agg_cm += cm
        mcc_ = matthews_corrcoef(ybv, vp)
        acc_ = (cm[0, 0] + cm[1, 1]) / cm.sum()
        f1_ = f1_score(ybv, vp)
        sn_ = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        sp_ = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        auc_ = roc_auc_score(ybv, pen)
        fold_scores.append(pen.astype(np.float32))
        fold_labels.append(ybv.astype(np.int64))
        fold_aucs.append(auc_)
        fold_metrics.append([acc_, f1_, sn_, sp_, mcc_, auc_])
        print(f"  Fold {fi+1}: ACC={acc_:.4f} F1={f1_:.4f} Sn={sn_:.4f} Sp={sp_:.4f} "
              f"MCC={mcc_:+.4f} AUC={auc_:.4f}  CM=({cm[0,0]},{cm[0,1]},{cm[1,0]},{cm[1,1]})  FD={fd}")
    TN, FP, FN, TP = agg_cm[0, 0], agg_cm[0, 1], agg_cm[1, 0], agg_cm[1, 1]
    total = TP + TN + FP + FN
    agg_acc = (TP + TN) / total if total > 0 else 0
    agg_f1 = f1_score([0]*TN+[0]*FP+[1]*FN+[1]*TP, [0]*TN+[1]*FP+[0]*FN+[1]*TP)
    agg_sn = TP / (TP + FN) if (TP + FN) > 0 else 0
    agg_sp = TN / (TN + FP) if (TN + FP) > 0 else 0
    agg_mcc = matthews_corrcoef([0]*TN+[0]*FP+[1]*FN+[1]*TP, [0]*TN+[1]*FP+[0]*FN+[1]*TP)
    # pooled AUC: computed over the concatenated fold scores/labels (consistent with the
    # aggregated confusion matrix, rather than averaging the per-fold AUCs)
    all_y = np.concatenate(fold_labels); all_s = np.concatenate(fold_scores)
    agg_auc = roc_auc_score(all_y, all_s) if len(np.unique(all_y)) > 1 else 0.5
    print(f"  AGG[{label}]: ACC={agg_acc:.4f} F1={agg_f1:.4f} Sn={agg_sn:.4f} Sp={agg_sp:.4f} "
          f"MCC={agg_mcc:+.4f} AUC={agg_auc:.4f} (fold-AUC mean={np.mean(fold_aucs):.4f}) "
          f"({time.time()-t0_total:.0f}s)")
    return {'label': label, 'ACC': agg_acc, 'F1': agg_f1, 'Sn': agg_sn, 'Sp': agg_sp,
            'MCC': agg_mcc, 'AUC': agg_auc, 'CM': (TN, FP, FN, TP), 'fold_fds': fold_fds,
            'fold_scores': fold_scores, 'fold_labels': fold_labels,
            'fold_aucs': fold_aucs, 'fold_metrics': fold_metrics}

# ---- main flow ----
print("\n" + "#" * 70)
print(f"# MRMD3.0 in-fold feature selection (variant={VARIANT}, prefilter={PREFILTER})")
print(f"# Input features: {HSTACK_ORDER}  raw={Xa_all.shape[1]}d")
print("#" * 70)
print(f"# test_mode={test_mode}  N_FOLDS={N_FOLDS}  EA={EA}  EB={EB}")

t0_total = time.time()
label = f"CV-{VARIANT}"
res = run_training_insidecv(Xa_all, Xb_all, ya, yb, label)
res['elapsed'] = time.time() - t0_total

mode_str = 'TEST (2-fold)' if test_mode else 'FULL (5-fold)'
print(f"\n{'='*60}")
print(f"Done in {time.time()-t0_total:.0f}s  [{mode_str}]")

# ═══════════════════ results CSV ═══════════════════
TN, FP, FN, TP = res['CM']
df = pd.DataFrame([{
    'FD': res['fold_fds'][0] if res['fold_fds'] else K_USE,
    'ACC': round(res['ACC'], 4), 'F1': round(res['F1'], 4),
    'Sn': round(res['Sn'], 4), 'Sp': round(res['Sp'], 4),
    'MCC': round(res['MCC'], 4), 'AUC': round(res['AUC'], 4),
    'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN),
    'Elapsed_s': round(res['elapsed'], 1),
}])
out_csv = os.path.join(SCRIPT_DIR, 'main_results.csv')
df.to_csv(out_csv, index=False, quoting=1)
print(f"\nCSV saved: {out_csv}")

# ═══════════════════ per-fold npz (scores/labels/metrics, for reproduction and fold-level analysis) ═══════════════════
npz_path = os.path.join(SCRIPT_DIR, 'main_results.npz')
np.savez(
    npz_path,
    label=np.array([res['label']]),
    accs=np.array([res['ACC']]),
    f1s=np.array([res['F1']]),
    sns=np.array([res['Sn']]),
    sps=np.array([res['Sp']]),
    mccs=np.array([res['MCC']]),
    aucs=np.array([res['AUC']]),
    cms=np.array([list(res['CM'])]),
    fold_fds=np.array([res['fold_fds']], dtype=object),
    fold_scores=np.array([res['fold_scores']], dtype=object),
    fold_labels=np.array([res['fold_labels']], dtype=object),
    fold_aucs=np.array([res['fold_aucs']], dtype=object),
    fold_metrics=np.array([res['fold_metrics']], dtype=object),
)
print(f"npz saved: {npz_path}")

# ═══════════════════ per-fold detail CSV (full metrics per fold) ═══════════════════
fold_rows = []
for fi, m in enumerate(res.get('fold_metrics', []), start=1):
    acc_, f1_, sn_, sp_, mcc_, auc_ = m
    fold_rows.append({'Fold': fi,
                      'ACC': round(float(acc_), 4), 'F1': round(float(f1_), 4),
                      'Sn': round(float(sn_), 4), 'Sp': round(float(sp_), 4),
                      'MCC': round(float(mcc_), 4), 'AUC': round(float(auc_), 4)})
fold_csv = os.path.join(SCRIPT_DIR, 'main_fold_metrics.csv')
pd.DataFrame(fold_rows).to_csv(fold_csv, index=False, quoting=1)
print(f"fold-metrics CSV saved: {fold_csv}  ({len(fold_rows)} rows)")

print(df.to_string(index=False))
print("Done.")
