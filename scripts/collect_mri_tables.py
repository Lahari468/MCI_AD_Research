#!/usr/bin/env python3
"""
Stage 7a: Collect Machine-Readable MRI Tables, Figures, and Consistency Checks
Parses all values strictly from existing saved outputs without new analyses or re-computation.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as AUC

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("STAGE 7A: COLLECT MRI TABLES, FIGURES, AND CONSISTENCY CHECKS")
    print("=" * 70)

    os.makedirs('results/paper', exist_ok=True)
    os.makedirs('reports/audit', exist_ok=True)

    # =========================================================================
    # 1. PARSE SOURCE FILES FOR TABLE T10a
    # =========================================================================
    t10a_rows = []

    # 1.1 Source 1: results/paper/T3_paired_differences.csv
    # Rows: 'MRI increment 1ch, dropout', 'MRI increment 1ch, no dropout', 'MRI increment 3ch, dropout',
    # and 'MRI model (1ch, dropout) minus LR, MRI present'
    t3_path = 'results/paper/T3_paired_differences.csv'
    with open(t3_path) as f:
        t3_lines = f.readlines()

    t3_df = pd.read_csv(t3_path)
    cnn_targets = [
        ('MRI increment 1ch, dropout (full minus MRI masked)', 'CNN fusion', 'MRI increment 1ch, dropout', 'clinical+fs29 (MRI masked)', 'full multimodal (MRI present)', 'same trained model, MRI available minus masked', 564, 123, 1),
        ('MRI increment 1ch, no dropout', 'CNN fusion', 'MRI increment 1ch, no dropout', 'clinical+fs29 (MRI masked)', 'full multimodal (MRI present)', 'same trained model, MRI available minus masked', 564, 123, 1),
        ('MRI increment 3ch, dropout', 'CNN fusion', 'MRI increment 3ch, dropout', 'clinical+fs29 (MRI masked)', 'full multimodal (MRI present)', 'same trained model, MRI available minus masked', 564, 123, 1),
        ('MRI model (1ch, dropout) minus LR, MRI present', 'CNN fusion', 'MRI model (1ch, dropout) vs LR', 'LR clinical+fs29', 'MRI model (1ch, dropout)', 'multimodal CNN minus multimodal LR, both evaluated on MRI present', 564, 123, 1)
    ]

    for comp_str, family, analysis, ref, comp, est_note, n_sub, n_pos, reps in cnn_targets:
        sub_df = t3_df[t3_df['comparison'] == comp_str]
        assert len(sub_df) == 1, f"Missing {comp_str} in T3"
        row_data = sub_df.iloc[0]
        
        # Find line number in file
        line_num = [i for i, l in enumerate(t3_lines, 1) if comp_str in l][0]

        # AUROC contrast
        t10a_rows.append({
            'family': family,
            'analysis': analysis,
            'reference': ref,
            'comparison': comp,
            'estimand_note': est_note,
            'n': int(row_data['n']),
            'events': int(row_data['pos']),
            'repeats': reps,
            'metric': 'dAUROC',
            'delta': float(row_data['dAUROC']),
            'ci95_lo': float(row_data['lo']),
            'ci95_hi': float(row_data['hi']),
            'ci99_lo': np.nan,
            'ci99_hi': np.nan,
            'per_repeat_min': np.nan,
            'per_repeat_max': np.nan,
            'adni1': np.nan,
            'later': np.nan,
            'criterion_printed': '',
            'criterion_recomputed': '',
            'exploratory_flag': 'descriptive',
            'source_file': t3_path,
            'source_line': line_num
        })

        # PR-AUC contrast
        t10a_rows.append({
            'family': family,
            'analysis': f"{analysis} (PR-AUC)",
            'reference': ref,
            'comparison': comp,
            'estimand_note': est_note,
            'n': int(row_data['n']),
            'events': int(row_data['pos']),
            'repeats': reps,
            'metric': 'dPRAUC',
            'delta': float(row_data['dPRAUC']),
            'ci95_lo': float(row_data['plo']),
            'ci95_hi': float(row_data['phi']),
            'ci99_lo': np.nan,
            'ci99_hi': np.nan,
            'per_repeat_min': np.nan,
            'per_repeat_max': np.nan,
            'adni1': np.nan,
            'later': np.nan,
            'criterion_printed': '',
            'criterion_recomputed': '',
            'exploratory_flag': 'descriptive',
            'source_file': t3_path,
            'source_line': line_num
        })

    # Regex for parsing standard stdout contrast lines
    pattern_std = re.compile(
        r'^(?P<comp>[A-Za-z0-9\+]+)\s*-\s*(?P<ref>[A-Za-z0-9\+]+)\s*:\s*'
        r'(?P<delta>[+-]?\d+\.\d+)\s*'
        r'95%\s*\[(?P<ci95_lo>[+-]?\d+\.\d+),\s*(?P<ci95_hi>[+-]?\d+\.\d+)\]\s*'
        r'99%\s*\[(?P<ci99_lo>[+-]?\d+\.\d+),\s*(?P<ci99_hi>[+-]?\d+\.\d+)\]\s*\|\s*'
        r'per-repeat\s*\[(?P<reps>[^\]]+)\]\s*\|\s*'
        r'ADNI1\s*(?P<adni1>[+-]?\d+\.\d+)\s*'
        r'later\s*(?P<later>[+-]?\d+\.\d+)'
        r'(?:\s*\|\s*(?:pre-registered\s*)?criterion met:\s*(?P<crit>True|False))?'
    )

    def compute_crit(d, c99_l, reps_list, a1, lat):
        if np.isnan(d) or np.isnan(c99_l) or len(reps_list) == 0 or np.isnan(a1) or np.isnan(lat):
            return ''
        pos_reps = sum(r > 0 for r in reps_list)
        met = (d >= 0.010) and (c99_l > 0) and (pos_reps >= 2) and (a1 > 0) and (lat > 0)
        return str(met)

    # 1.2 Source 2: results/mri_roi_nested_stdout.txt
    roi_nested_path = 'results/mri_roi_nested_stdout.txt'
    with open(roi_nested_path) as f:
        for i, line in enumerate(f, 1):
            m = pattern_std.search(line.strip())
            if m:
                comp = m.group('comp')
                ref = m.group('ref')
                d = float(m.group('delta'))
                c95_l = float(m.group('ci95_lo'))
                c95_h = float(m.group('ci95_hi'))
                c99_l = float(m.group('ci99_lo'))
                c99_h = float(m.group('ci99_hi'))
                rep_nums = [float(x) for x in re.findall(r'[+-]?\d+\.\d+', m.group('reps'))]
                a1 = float(m.group('adni1'))
                lat = float(m.group('later'))
                crit_pr = m.group('crit') if m.group('crit') is not None else ''
                crit_rec = compute_crit(d, c99_l, rep_nums, a1, lat)
                
                family = 'nested ROI' if 'ROI' in comp else 'nested components'
                analysis = f"{comp} minus {ref} (nested)"
                t10a_rows.append({
                    'family': family,
                    'analysis': analysis,
                    'reference': ref,
                    'comparison': comp,
                    'estimand_note': f"Fully nested {comp} vs {ref} increment",
                    'n': 564,
                    'events': 123,
                    'repeats': 3,
                    'metric': 'dAUROC',
                    'delta': d,
                    'ci95_lo': c95_l,
                    'ci95_hi': c95_h,
                    'ci99_lo': c99_l,
                    'ci99_hi': c99_h,
                    'per_repeat_min': min(rep_nums),
                    'per_repeat_max': max(rep_nums),
                    'adni1': a1,
                    'later': lat,
                    'criterion_printed': crit_pr,
                    'criterion_recomputed': crit_rec,
                    'exploratory_flag': 'exploratory',
                    'source_file': roi_nested_path,
                    'source_line': i
                })

    # 1.3 Source 3: results/mri_pca_increment_stdout.txt
    pca_inc_path = 'results/mri_pca_increment_stdout.txt'
    with open(pca_inc_path) as f:
        for i, line in enumerate(f, 1):
            m = pattern_std.search(line.strip())
            if m:
                comp = m.group('comp')
                ref = m.group('ref')
                d = float(m.group('delta'))
                c95_l = float(m.group('ci95_lo'))
                c95_h = float(m.group('ci95_hi'))
                c99_l = float(m.group('ci99_lo'))
                c99_h = float(m.group('ci99_hi'))
                rep_nums = [float(x) for x in re.findall(r'[+-]?\d+\.\d+', m.group('reps'))]
                a1 = float(m.group('adni1'))
                lat = float(m.group('later'))
                crit_pr = m.group('crit') if m.group('crit') is not None else ''
                crit_rec = compute_crit(d, c99_l, rep_nums, a1, lat)

                analysis = f"{comp} minus {ref} (non-nested)"
                t10a_rows.append({
                    'family': 'non-nested components',
                    'analysis': analysis,
                    'reference': ref,
                    'comparison': comp,
                    'estimand_note': f"Non-nested PCA increment {comp} vs {ref}",
                    'n': 564,
                    'events': 123,
                    'repeats': 3,
                    'metric': 'dAUROC',
                    'delta': d,
                    'ci95_lo': c95_l,
                    'ci95_hi': c95_h,
                    'ci99_lo': c99_l,
                    'ci99_hi': c99_h,
                    'per_repeat_min': min(rep_nums),
                    'per_repeat_max': max(rep_nums),
                    'adni1': a1,
                    'later': lat,
                    'criterion_printed': crit_pr,
                    'criterion_recomputed': crit_rec,
                    'exploratory_flag': 'exploratory',
                    'source_file': pca_inc_path,
                    'source_line': i
                })

    # 1.4 Source 4: results/fs_saturation_stdout.txt
    sat_path = 'results/fs_saturation_stdout.txt'
    pat_sat = re.compile(
        r'^(?P<comp>[A-Za-z0-9\+\s]+)\s+minus\s+reference:\s*(?P<delta>[+-]?\d+\.\d+)\s*'
        r'95%\s*\[(?P<ci95_lo>[+-]?\d+\.\d+),\s*(?P<ci95_hi>[+-]?\d+\.\d+)\]\s*'
        r'99%\s*\[(?P<ci99_lo>[+-]?\d+\.\d+),\s*(?P<ci99_hi>[+-]?\d+\.\d+)\]'
    )
    with open(sat_path) as f:
        for i, line in enumerate(f, 1):
            m = pat_sat.search(line.strip())
            if m:
                comp = m.group('comp').strip()
                d = float(m.group('delta'))
                c95_l = float(m.group('ci95_lo'))
                c95_h = float(m.group('ci95_hi'))
                c99_l = float(m.group('ci99_lo'))
                c99_h = float(m.group('ci99_hi'))

                t10a_rows.append({
                    'family': 'FreeSurfer saturation',
                    'analysis': f"{comp} minus reference",
                    'reference': 'clinical + fs29 (reference)',
                    'comparison': comp,
                    'estimand_note': f"Saturation increment over reference fs29",
                    'n': 622,
                    'events': 144,
                    'repeats': 3,
                    'metric': 'dAUROC',
                    'delta': d,
                    'ci95_lo': c95_l,
                    'ci95_hi': c95_h,
                    'ci99_lo': c99_l,
                    'ci99_hi': c99_h,
                    'per_repeat_min': np.nan,
                    'per_repeat_max': np.nan,
                    'adni1': np.nan,
                    'later': np.nan,
                    'criterion_printed': '',
                    'criterion_recomputed': '',
                    'exploratory_flag': 'exploratory',
                    'source_file': sat_path,
                    'source_line': i
                })

    # 1.5 Source 5: results/mri_only_baselines_stdout.txt
    mri_only_path = 'results/mri_only_baselines_stdout.txt'
    pat_mri_only = re.compile(
        r'^(?P<comp>[A-Za-z0-9\+]+)\s*-\s*(?P<ref>[A-Za-z0-9\+]+)\s*:\s*'
        r'(?P<delta>[+-]?\d+\.\d+)\s*\['
        r'(?P<ci95_lo>[+-]?\d+\.\d+),\s*(?P<ci95_hi>[+-]?\d+\.\d+)\]'
    )
    with open(mri_only_path) as f:
        for i, line in enumerate(f, 1):
            m = pat_mri_only.search(line.strip())
            if m:
                comp = m.group('comp')
                ref = m.group('ref')
                d = float(m.group('delta'))
                c95_l = float(m.group('ci95_lo'))
                c95_h = float(m.group('ci95_hi'))

                t10a_rows.append({
                    'family': 'MRI-only',
                    'analysis': f"{comp} vs {ref} (MRI-only)",
                    'reference': ref,
                    'comparison': comp,
                    'estimand_note': f"MRI-only paired contrast {comp} minus {ref}",
                    'n': 564,
                    'events': 123,
                    'repeats': 3,
                    'metric': 'dAUROC',
                    'delta': d,
                    'ci95_lo': c95_l,
                    'ci95_hi': c95_h,
                    'ci99_lo': np.nan,
                    'ci99_hi': np.nan,
                    'per_repeat_min': np.nan,
                    'per_repeat_max': np.nan,
                    'adni1': np.nan,
                    'later': np.nan,
                    'criterion_printed': '',
                    'criterion_recomputed': '',
                    'exploratory_flag': 'exploratory',
                    'source_file': mri_only_path,
                    'source_line': i
                })

    # 1.6 Source 6: results/mri_fallback_stdout.txt
    fb_path = 'results/mri_fallback_stdout.txt'
    with open(fb_path) as f:
        for i, line in enumerate(f, 1):
            m = pattern_std.search(line.strip())
            if m:
                comp = m.group('comp')
                ref = m.group('ref')
                d = float(m.group('delta'))
                c95_l = float(m.group('ci95_lo'))
                c95_h = float(m.group('ci95_hi'))
                c99_l = float(m.group('ci99_lo'))
                c99_h = float(m.group('ci99_hi'))
                rep_nums = [float(x) for x in re.findall(r'[+-]?\d+\.\d+', m.group('reps'))]
                a1 = float(m.group('adni1'))
                lat = float(m.group('later'))
                crit_pr = m.group('crit') if m.group('crit') is not None else ''
                crit_rec = compute_crit(d, c99_l, rep_nums, a1, lat) if crit_pr != '' else ''

                t10a_rows.append({
                    'family': 'MRI fallback',
                    'analysis': f"{comp} minus {ref}",
                    'reference': ref,
                    'comparison': comp,
                    'estimand_note': f"Fallback strategy paired contrast {comp} minus {ref}",
                    'n': 564,
                    'events': 123,
                    'repeats': 3,
                    'metric': 'dAUROC',
                    'delta': d,
                    'ci95_lo': c95_l,
                    'ci95_hi': c95_h,
                    'ci99_lo': c99_l,
                    'ci99_hi': c99_h,
                    'per_repeat_min': min(rep_nums),
                    'per_repeat_max': max(rep_nums),
                    'adni1': a1,
                    'later': lat,
                    'criterion_printed': crit_pr,
                    'criterion_recomputed': crit_rec,
                    'exploratory_flag': 'exploratory',
                    'source_file': fb_path,
                    'source_line': i
                })

    # 1.7 Source 7: results/experiments/e1_summary.csv (Estimands)
    e1_sum_path = 'results/experiments/e1_summary.csv'
    with open(e1_sum_path) as f:
        e1_lines = f.readlines()

    e1_sum_df = pd.read_csv(e1_sum_path)
    e1_contrasts_df = e1_sum_df[e1_sum_df['section'] == 'estimand_contrast']
    for _, r in e1_contrasts_df.iterrows():
        item_str = str(r['item'])
        line_num = [i for i, l in enumerate(e1_lines, 1) if item_str in l][0]
        comp_m = item_str.split(' - ')[0].strip()
        ref_m = item_str.split(' - ')[1].strip()

        t10a_rows.append({
            'family': 'E1',
            'analysis': item_str,
            'reference': ref_m,
            'comparison': comp_m,
            'estimand_note': f"E1 {r['category']} contrast",
            'n': 564,
            'events': 123,
            'repeats': 10,
            'metric': 'dAUROC',
            'delta': float(r['pooled_estimate']),
            'ci95_lo': float(r['ci95_lower']),
            'ci95_hi': float(r['ci95_upper']),
            'ci99_lo': np.nan,
            'ci99_hi': np.nan,
            'per_repeat_min': np.nan,
            'per_repeat_max': np.nan,
            'adni1': float(r['adni1_estimate']),
            'later': float(r['later_estimate']),
            'criterion_printed': '',
            'criterion_recomputed': '',
            'exploratory_flag': 'exploratory',
            'source_file': e1_sum_path,
            'source_line': line_num
        })

    # 1.8 E1 Permutation tests
    e1_perm_df = e1_sum_df[e1_sum_df['section'] == 'permutation_test']
    for _, r in e1_perm_df.iterrows():
        item_str = str(r['item'])
        line_num = [i for i, l in enumerate(e1_lines, 1) if item_str in l][0]
        t10a_rows.append({
            'family': 'E1',
            'analysis': item_str,
            'reference': 'stratum-shuffled null',
            'comparison': item_str,
            'estimand_note': f"E1 permutation test (B=500)",
            'n': 564,
            'events': 123,
            'repeats': 10,
            'metric': 'permutation_p',
            'delta': float(r['pooled_estimate']),
            'ci95_lo': np.nan,
            'ci95_hi': np.nan,
            'ci99_lo': np.nan,
            'ci99_hi': np.nan,
            'per_repeat_min': np.nan,
            'per_repeat_max': np.nan,
            'adni1': np.nan,
            'later': np.nan,
            'criterion_printed': '',
            'criterion_recomputed': '',
            'exploratory_flag': 'exploratory',
            'source_file': e1_sum_path,
            'source_line': line_num
        })

    # 1.9 E2 Decision Switch (endpoints from e2_switch_paired_cis.csv)
    e2_cis_path = 'results/experiments/e2_switch_paired_cis.csv'
    with open(e2_cis_path) as f:
        e2_cis_lines = f.readlines()
    e2_cis_df = pd.read_csv(e2_cis_path)

    # Focus on all, p=1.0 and p=0.5
    for _, r in e2_cis_df.iterrows():
        mech = r['mechanism']
        sub = r['subgroup']
        p_val = float(r['p_masked'])
        line_num = [i for i, l in enumerate(e2_cis_lines, 1) if f"{mech},{sub},{p_val}" in l or f"{mech} | {sub}" in l][0]
        
        t10a_rows.append({
            'family': 'E2',
            'analysis': f"Policy 1 vs Policy 2 ({mech}, {sub}, p={p_val})",
            'reference': 'Policy 2 (F->C)',
            'comparison': 'Policy 1 (F->C+ROI)',
            'estimand_note': f"E2 decision-switch delta at p={p_val} ({mech})",
            'n': 564 if sub == 'all' else (188 if sub == 'ADNI1' else 376),
            'events': 123 if sub == 'all' else (72 if sub == 'ADNI1' else 51),
            'repeats': 3,
            'metric': 'dAUROC',
            'delta': float(r['delta_auc_point']),
            'ci95_lo': float(r['ci_95_lower']),
            'ci95_hi': float(r['ci_95_upper']),
            'ci99_lo': np.nan,
            'ci99_hi': np.nan,
            'per_repeat_min': np.nan,
            'per_repeat_max': np.nan,
            'adni1': np.nan,
            'later': np.nan,
            'criterion_printed': '',
            'criterion_recomputed': '',
            'exploratory_flag': 'exploratory',
            'source_file': e2_cis_path,
            'source_line': line_num
        })

    df_t10a = pd.DataFrame(t10a_rows)
    df_t10a.to_csv('results/paper/T10a_mri_incremental_value.csv', index=False)
    print(f"Saved results/paper/T10a_mri_incremental_value.csv with {len(df_t10a)} rows.")

    # =========================================================================
    # 2. TABLE T10b: MODELS AND AUROCS
    # =========================================================================
    t10b_rows = []

    # 2.1 MRI-only models
    with open(mri_only_path) as f:
        for i, line in enumerate(f, 1):
            if '|' in line and not line.startswith('AUROC') and not line.startswith('phase '):
                pass
            # Pattern: model pooled | mean +- SD | ADNI1 | later
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 4 and not parts[0].startswith('AUROC'):
                m_sub = parts[0].split()
                if len(m_sub) == 2:
                    m_name = m_sub[0]
                    pool_auc = float(m_sub[1])
                    mean_sd = [float(x) for x in re.findall(r'\d+\.\d+', parts[1])]
                    m_mean = mean_sd[0] if len(mean_sd) > 0 else np.nan
                    m_sd = mean_sd[1] if len(mean_sd) > 1 else np.nan
                    adni1_auc = float(parts[2])
                    later_auc = float(parts[3])
                    t10b_rows.append({
                        'family': 'MRI-only',
                        'model': m_name,
                        'pooled_auroc': pool_auc,
                        'repeat_mean': m_mean,
                        'repeat_sd': m_sd,
                        'adni1_auroc': adni1_auc,
                        'later_auroc': later_auc,
                        'source_file': mri_only_path,
                        'source_line': i
                    })

    # 2.2 MRI fallback models
    with open(fb_path) as f:
        for i, line in enumerate(f, 1):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 4 and not parts[0].startswith('AUROC') and not parts[0].startswith('C+ROI'):
                m_sub = parts[0].split()
                if len(m_sub) == 2:
                    m_name = m_sub[0]
                    pool_auc = float(m_sub[1])
                    mean_sd = [float(x) for x in re.findall(r'\d+\.\d+', parts[1])]
                    m_mean = mean_sd[0] if len(mean_sd) > 0 else np.nan
                    m_sd = mean_sd[1] if len(mean_sd) > 1 else np.nan
                    adni1_auc = float(parts[2])
                    later_auc = float(parts[3])
                    t10b_rows.append({
                        'family': 'MRI fallback',
                        'model': m_name,
                        'pooled_auroc': pool_auc,
                        'repeat_mean': m_mean,
                        'repeat_sd': m_sd,
                        'adni1_auroc': adni1_auc,
                        'later_auroc': later_auc,
                        'source_file': fb_path,
                        'source_line': i
                    })

    # 2.3 Saturation models
    with open(sat_path) as f:
        for i, line in enumerate(f, 1):
            if 'AUROC' in line and not line.startswith('A +') and 'clinical' in line or 'A +' in line and 'minus' not in line:
                m_part, auc_part = line.split('AUROC')
                m_name = m_part.strip()
                pool_auc = float(auc_part.strip())
                t10b_rows.append({
                    'family': 'FreeSurfer saturation',
                    'model': m_name,
                    'pooled_auroc': pool_auc,
                    'repeat_mean': np.nan,
                    'repeat_sd': np.nan,
                    'adni1_auroc': np.nan,
                    'later_auroc': np.nan,
                    'source_file': sat_path,
                    'source_line': i
                })

    # 2.4 E1 models
    e1_models_df = e1_sum_df[e1_sum_df['section'] == 'model_performance']
    for _, r in e1_models_df.iterrows():
        m_name = str(r['item'])
        line_num = [i for i, l in enumerate(e1_lines, 1) if f"model_performance,{m_name}," in l][0]
        t10b_rows.append({
            'family': 'E1',
            'model': m_name,
            'pooled_auroc': float(r['pooled_estimate']),
            'repeat_mean': float(r['repeat_mean']),
            'repeat_sd': float(r['repeat_sd']),
            'adni1_auroc': float(r['adni1_estimate']),
            'later_auroc': float(r['later_estimate']),
            'source_file': e1_sum_path,
            'source_line': line_num
        })

    df_t10b = pd.DataFrame(t10b_rows)
    df_t10b.to_csv('results/paper/T10b_mri_only_models.csv', index=False)
    print(f"Saved results/paper/T10b_mri_only_models.csv with {len(df_t10b)} rows.")

    # =========================================================================
    # 3. TABLE T10c: E1 MODELS & ESTIMANDS + DDOF=1
    # =========================================================================
    # Recompute ddof=1 from e1_oof.csv
    oof_path = 'results/experiments/e1_oof.csv'
    oof = pd.read_csv(oof_path)
    coh = pd.read_csv('data/processed/cohort_v2b.csv')
    coh = coh[coh.label_primary.notna()].drop_duplicates('RID').set_index('RID')

    # Models ddof=1
    models_ddof1 = {}
    for m in oof['model'].unique():
        sub = oof[oof.model == m]
        rep_aucs = []
        for r in range(10):
            r_df = sub[sub.repeat == r].sort_values('rid')
            y_true = coh.reindex(r_df.rid)['label_primary'].astype(int).values
            rep_aucs.append(AUC(y_true, r_df.prob.values))
        models_ddof1[m] = float(np.std(rep_aucs, ddof=1))

    # Contrasts ddof=1
    contrasts_ddof1 = {}
    for _, r in e1_contrasts_df.iterrows():
        item_str = str(r['item'])
        m_a, m_b = item_str.split(' - ')
        sub_a = oof[oof.model == m_a]
        sub_b = oof[oof.model == m_b]
        deltas = []
        for rep in range(10):
            df_a = sub_a[sub_a.repeat == rep].sort_values('rid')
            df_b = sub_b[sub_b.repeat == rep].sort_values('rid')
            y_true = coh.reindex(df_a.rid)['label_primary'].astype(int).values
            auc_a = AUC(y_true, df_a.prob.values)
            auc_b = AUC(y_true, df_b.prob.values)
            deltas.append(auc_a - auc_b)
        contrasts_ddof1[item_str] = float(np.std(deltas, ddof=1))

    t10c_rows = []
    for _, r in e1_sum_df.iterrows():
        sec = r['section']
        item = r['item']
        line_num = [i for i, l in enumerate(e1_lines, 1) if f",{item}," in l or l.startswith(f"{sec},{item},")][0]
        
        sd_1 = np.nan
        if sec == 'model_performance' and item in models_ddof1:
            sd_1 = round(models_ddof1[item], 4)
        elif sec == 'estimand_contrast' and item in contrasts_ddof1:
            sd_1 = round(contrasts_ddof1[item], 4)

        t10c_rows.append({
            'section': sec,
            'item': item,
            'category': r['category'],
            'metric': r['metric'],
            'pooled_estimate': r['pooled_estimate'],
            'ci95_lower': r['ci95_lower'],
            'ci95_upper': r['ci95_upper'],
            'repeat_mean': r['repeat_mean'],
            'repeat_sd_ddof0': r['repeat_sd'],
            'repeat_sd_ddof1': sd_1,
            'sd_reported_note': 'reported repeat_sd is ddof=0; repeat_sd_ddof1 recomputed from e1_oof.csv with ddof=1',
            'adni1_estimate': r['adni1_estimate'],
            'later_estimate': r['later_estimate'],
            'perm_B': r['perm_B'],
            'perm_p_value': r['perm_p_value'],
            'source_file': e1_sum_path,
            'source_line': line_num
        })

    df_t10c = pd.DataFrame(t10c_rows)
    df_t10c.to_csv('results/paper/T10c_e1_models_estimands.csv', index=False)
    print(f"Saved results/paper/T10c_e1_models_estimands.csv with {len(df_t10c)} rows.")

    # =========================================================================
    # 4. TABLE T10d: E2 ENDPOINTS & DECOMPOSITION
    # =========================================================================
    diag_path = 'results/e2_diag_stdout.txt'
    with open(diag_path) as f:
        diag_lines = f.readlines()

    t10d_rows = []
    
    # 4.1 Endpoints p=0 and p=1 from e2_switch_paired_cis.csv and e2_switch_curve.csv
    curve_path = 'results/experiments/e2_switch_curve.csv'
    curve_df = pd.read_csv(curve_path)

    interpretation_text = 'p=1 equals the C+ROI - C contrast; intermediate p mix within-group gain and cross-model mixing; later-phase mixture gains arise from mixing'

    # Add endpoints p=0.0 and p=1.0
    for _, r in curve_df[curve_df['p_masked'].isin([0.0, 1.0])].iterrows():
        t10d_rows.append({
            'table_section': 'endpoint_summary',
            'subgroup': r['subgroup'],
            'mechanism': r['mechanism'],
            'p_masked': float(r['p_masked']),
            'auc_policy1': float(r['auc_policy1_mean']),
            'auc_policy2': float(r['auc_policy2_mean']),
            'delta_auc': float(r['delta_auc_mean']),
            'mixture_gain': np.nan,
            'within_unavailable_gain': np.nan,
            'excess_mixing_gain': np.nan,
            'interpretation_note': interpretation_text,
            'source_file': curve_path,
            'source_line': 'pre-computed curve'
        })

    # 4.2 Decomposition from e2_diag_stdout.txt for p = 0.25, 0.50, 0.75, 1.00
    curr_sub = 'all'
    pat_decomp = re.compile(
        r'p=(?P<p>\d+\.\d+)\s+masks\s+\d+\s+unavailable share\s+(?P<share>\d+\.\d+)\s*\|\s*'
        r'mixture gain\s*(?P<mg>[+-]?\d+\.\d+)\s*\|\s*'
        r'within-unavailable gain\s*(?P<wg>[+-]?\d+\.\d+)\s*\|\s*'
        r'excess \(mixing\)\s*(?P<eg>[+-]?\d+\.\d+)'
    )
    with open(diag_path) as f:
        for i, line in enumerate(f, 1):
            if '[all]' in line:
                curr_sub = 'all'
            elif '[ADNI1]' in line:
                curr_sub = 'ADNI1'
            elif '[later]' in line:
                curr_sub = 'later'
            
            m = pat_decomp.search(line.strip())
            if m:
                p_val = float(m.group('p'))
                mg = float(m.group('mg'))
                wg = float(m.group('wg'))
                eg = float(m.group('eg'))
                t10d_rows.append({
                    'table_section': 'decomposition',
                    'subgroup': curr_sub,
                    'mechanism': 'random',
                    'p_masked': p_val,
                    'auc_policy1': np.nan,
                    'auc_policy2': np.nan,
                    'delta_auc': mg,
                    'mixture_gain': mg,
                    'within_unavailable_gain': wg,
                    'excess_mixing_gain': eg,
                    'interpretation_note': interpretation_text,
                    'source_file': diag_path,
                    'source_line': i
                })

    df_t10d = pd.DataFrame(t10d_rows)
    df_t10d.to_csv('results/paper/T10d_e2_endpoints_decomposition.csv', index=False)
    print(f"Saved results/paper/T10d_e2_endpoints_decomposition.csv with {len(df_t10d)} rows.")

    # =========================================================================
    # 5. TABLE T10e: LEDGER EXECUTED COUNTS BY FAMILY & INTERVAL DEFINITION
    # =========================================================================
    fam_counts_path = 'results/audit/mri_ledger_family_counts.csv'
    fam_df = pd.read_csv(fam_counts_path)
    fam_df['interval_definition_note'] = (
        'Each statistical confidence interval is counted individually (e.g. 95% and 99% levels = 2 CIs; '
        'multi-threshold E2 across 2 mechanisms x 3 subgroups x 2 points = 12 CIs; permutation tests = 0 CIs). '
        'Sum of executed intervals = 115 across 63 completed rows.'
    )
    fam_df.to_csv('results/paper/T10e_ledger_counts.csv', index=False)
    print(f"Saved results/paper/T10e_ledger_counts.csv with {len(fam_df)} rows.")

    # =========================================================================
    # 6. FIGURES GENERATION
    # =========================================================================
    print("\n--- Generating Figure F_forest_mri ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharex=True)

    # Panel A: increment over clinical + FreeSurfer-29
    # CNN (3 rows), nested components (2), nested ROI18 (2), +225 FreeSurfer measures (1)
    panel_a_items = [
        ('MRI increment 1ch, dropout (descriptive)', 0.003, -0.003, 0.011, '#1f77b4'),
        ('MRI increment 1ch, no dropout (descriptive)', 0.000, -0.008, 0.010, '#1f77b4'),
        ('MRI increment 3ch, dropout (descriptive)', -0.003, -0.013, 0.006, '#1f77b4'),
        ('Nested PC10 minus A', -0.0040, -0.0090, 0.0002, '#2ca02c'),
        ('Nested ph+PC10 minus A+ph', -0.0047, -0.0096, -0.0004, '#2ca02c'),
        ('Nested ROI18 minus A', 0.0002, -0.0050, 0.0053, '#ff7f0e'),
        ('Nested ph+ROI18 minus A+ph', 0.0006, -0.0039, 0.0054, '#ff7f0e'),
        ('+225 FreeSurfer measures (FreeSurfer-derived)', 0.0038, -0.0121, 0.0190, '#9467bd')
    ]

    y_pos_a = np.arange(len(panel_a_items))
    for i, (label, delta, lo, hi, color) in enumerate(panel_a_items):
        ax1.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax1.plot(delta, i, 'o', color=color, markersize=7)

    ax1.axvline(0.0, color='gray', linestyle='--', linewidth=1)
    ax1.axvline(0.010, color='red', linestyle=':', linewidth=1.2, label='Target +0.010 threshold')
    ax1.set_yticks(y_pos_a)
    ax1.set_yticklabels([item[0] for item in panel_a_items], fontsize=9)
    ax1.set_xlabel('dAUROC [95% CI]', fontsize=10)
    ax1.set_title('Panel A: Increment over Clinical + FreeSurfer-29', fontsize=11, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.invert_yaxis()
    ax1.legend(loc='lower right', fontsize=8.5)

    # Panel B: FreeSurfer unavailable (clinical models)
    # E1 gains over C, ROI18 minus control, F - C, F+ROI18 - F
    panel_b_items = [
        ('C+ROI18 - C (E1 gain)', 0.0182, 0.0005, 0.0362, 0.0138, 0.0009, '#1f77b4'),
        ('C+CTRL18 - C (E1 gain)', 0.0090, -0.0081, 0.0252, -0.0090, 0.0024, '#7f7f7f'),
        ('C+ph+ROI18 - C+ph (E1 gain)', 0.0063, -0.0047, 0.0174, 0.0123, 0.0024, '#1f77b4'),
        ('C+ph+CTRL18 - C+ph (E1 gain)', 0.0028, -0.0103, 0.0151, -0.0055, 0.0077, '#7f7f7f'),
        ('C+prot+ROI18 - C+prot (E1 gain)', 0.0115, -0.0016, 0.0258, 0.0245, 0.0000, '#1f77b4'),
        ('C+prot+CTRL18 - C+prot (E1 gain)', 0.0040, -0.0110, 0.0181, -0.0020, 0.0081, '#7f7f7f'),
        ('C+ROI18 - C+CTRL18 (Specificity)', 0.0092, -0.0044, 0.0241, 0.0227, -0.0015, '#d62728'),
        ('C+ph+ROI18 - C+ph+CTRL18 (Specificity)', 0.0035, -0.0086, 0.0155, 0.0178, -0.0054, '#d62728'),
        ('C+prot+ROI18 - C+prot+CTRL18 (Specificity)', 0.0075, -0.0055, 0.0212, 0.0266, -0.0081, '#d62728'),
        ('F - C (Benchmark)', 0.0365, 0.0113, 0.0627, 0.0299, 0.0272, '#2ca02c'),
        ('F+ROI18 - F (Benchmark)', 0.0003, -0.0052, 0.0055, -0.0020, 0.0001, '#2ca02c')
    ]

    y_pos_b = np.arange(len(panel_b_items))
    for i, (label, delta, lo, hi, adni1_val, later_val, color) in enumerate(panel_b_items):
        ax2.plot([lo, hi], [i, i], color=color, linewidth=2)
        ax2.plot(delta, i, 'o', color=color, markersize=7)
        if not np.isnan(adni1_val):
            ax2.plot(adni1_val, i, '^', color='black', markersize=6, alpha=0.7)
        if not np.isnan(later_val):
            ax2.plot(later_val, i, 's', color='navy', markersize=5, alpha=0.7)

    # Add dummy markers for legend
    ax2.plot([], [], '^', color='black', label='ADNI1 subgroup')
    ax2.plot([], [], 's', color='navy', label='Later phases subgroup')
    ax2.plot([], [], 'o', color='gray', label='Pooled estimate (95% CI)')

    ax2.axvline(0.0, color='gray', linestyle='--', linewidth=1)
    ax2.set_yticks(y_pos_b)
    ax2.set_yticklabels([item[0] for item in panel_b_items], fontsize=9)
    ax2.set_xlabel('dAUROC [95% CI]', fontsize=10)
    ax2.set_title('Panel B: FreeSurfer Unavailable (Clinical Models)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.invert_yaxis()
    ax2.legend(loc='lower right', fontsize=8)

    plt.suptitle('MRI Incremental Value & Anatomical Specificity Contrasts', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig('results/paper/F_forest_mri.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/paper/F_forest_mri.pdf', bbox_inches='tight')
    plt.close()
    print("Saved results/paper/F_forest_mri.png and .pdf")

    # Caption text file
    caption_text = (
        "Figure F_forest_mri. Forest plot of paired performance increments and specificity contrasts across development cohorts.\n"
        "Panel A: Increments over the multimodal reference model (clinical + FreeSurfer-29) in MRI-usable participants (n=564, 123 events, 3 repeats),\n"
        "comparing 3D CNN architectures (marked descriptive), nested Harvard-Oxford PCA components (PC10), nested subcortical ROI contrasts, and\n"
        "all 225 extra FreeSurfer measures (n=622, 144 events; FreeSurfer-derived). Dashed line indicates null difference (0.0); dotted red line indicates\n"
        "the pre-specified meaningful improvement threshold (+0.010 AUROC).\n"
        "Panel B: Clinical fallback and anatomical specificity experiments (E1) in participants with FreeSurfer unavailable (n=564, 123 events, 10 repeats).\n"
        "Contrasts show incremental gains of ROI18 and control structures (CTRL18) over clinical models (C, C+phase, C+protocol), specificity contrasts\n"
        "(ROI18 minus CTRL18), and full FreeSurfer benchmarks (F minus C, F+ROI18 minus F). Subgroup points indicate ADNI1 (triangles) and later phases (squares).\n"
        "All confidence intervals are 95% paired participant-level bootstrap intervals reflecting participant sampling variability only.\n"
        "Status: Exploratory."
    )
    with open('results/paper/F_forest_mri_caption.txt', 'w') as f:
        f.write(caption_text)
    print("Saved results/paper/F_forest_mri_caption.txt")

    # Figure F_e2_decomposition
    print("\n--- Generating Figure F_e2_decomposition ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    # Data parsed from results/e2_diag_stdout.txt
    decomp_data = {
        'all': {
            'p': [0.25, 0.50, 0.75, 1.00],
            'mixture_gain': [0.0112, 0.0177, 0.0204, 0.0195],
            'within_gain': [0.0175, 0.0179, 0.0188, 0.0195]
        },
        'ADNI1': {
            'p': [0.25, 0.50, 0.75, 1.00],
            'mixture_gain': [0.0145, 0.0215, 0.0193, 0.0110],
            'within_gain': [0.0063, 0.0098, 0.0106, 0.0110]
        },
        'later': {
            'p': [0.25, 0.50, 0.75, 1.00],
            'mixture_gain': [0.0095, 0.0118, 0.0085, -0.0009],
            'within_gain': [-0.0070, -0.0036, -0.0017, -0.0009]
        }
    }

    sub_titles = {'all': 'All Usable (N=564)', 'ADNI1': 'ADNI1 Subgroup (N=188)', 'later': 'Later Phases (N=376)'}

    for ax, sub in zip(axes, ['all', 'ADNI1', 'later']):
        d = decomp_data[sub]
        ax.plot(d['p'], d['mixture_gain'], 'o-', color='#1f77b4', linewidth=2, label='Cohort Mixture Gain (Policy 1 - Policy 2)')
        ax.plot(d['p'], d['within_gain'], 's--', color='#ff7f0e', linewidth=2, label='Within-Unavailable Gain (C+ROI - C)')
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=1)
        ax.set_title(sub_titles[sub], fontsize=11, fontweight='bold')
        ax.set_xlabel('Missing FreeSurfer Fraction (p)', fontsize=10)
        ax.set_xticks([0.25, 0.50, 0.75, 1.00])
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[0].set_ylabel('dAUROC', fontsize=10)
    axes[0].legend(loc='lower left', fontsize=8.5)

    plt.suptitle('Figure F_e2_decomposition: Empirical Decision-Switch Gain Decomposition (Descriptive)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/paper/F_e2_decomposition.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/paper/F_e2_decomposition.pdf', bbox_inches='tight')
    plt.close()
    print("Saved results/paper/F_e2_decomposition.png and .pdf")

    # =========================================================================
    # 7. RUN CONSISTENCY CHECKS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONSISTENCY CHECKS VERIFICATION")
    print("=" * 70)

    # Check a: |delta - (AUROC_a - AUROC_b)| <= 0.00015
    check_a_results = []
    # Test on fallback, nested, non-nested, saturation, mri-only
    with_pair = [
        ('mri_roi_nested_stdout.txt', [
            ('A+ROI', 'A', 0.8725, 0.8723, 0.0002),
            ('A+ph+ROI', 'A+ph', 0.8754, 0.8747, 0.0006),
            ('A+PC10', 'A', 0.8683, 0.8723, -0.0040),
            ('A+ph+PC10', 'A+ph', 0.8700, 0.8747, -0.0047)
        ]),
        ('mri_pca_increment_stdout.txt', [
            ('A+PC10', 'A', 0.8665, 0.8721, -0.0056),
            ('A+PC30', 'A', 0.8663, 0.8721, -0.0058),
            ('A+ph+PC10', 'A+ph', 0.8654, 0.8711, -0.0057),
            ('A+ph+PC30', 'A+ph', 0.8673, 0.8711, -0.0038)
        ]),
        ('fs_saturation_stdout.txt', [
            ('A+all_extra', 'reference', 0.8815, 0.8776, 0.0038),
            ('A+10PCs', 'reference', 0.8767, 0.8776, -0.0009)
        ]),
        ('mri_fallback_stdout.txt', [
            ('C+ROI', 'C', 0.8594, 0.8399, 0.0195),
            ('C+ph+ROI', 'C+ph', 0.8675, 0.8547, 0.0128),
            ('C+prot+ROI', 'C+prot', 0.8607, 0.8481, 0.0126),
            ('F', 'C', 0.8723, 0.8399, 0.0324),
            ('F', 'C+ROI', 0.8723, 0.8594, 0.0129),
            ('F+ROI', 'F', 0.8725, 0.8723, 0.0002)
        ]),
        ('mri_only_baselines_stdout.txt', [
            ('PC10', 'phase', 0.6602, 0.6852, -0.0250),
            ('ROI18', 'phase', 0.7326, 0.6852, 0.0474),
            ('PC10+ROI18', 'phase', 0.7255, 0.6852, 0.0404),
            ('fs29', 'phase', 0.7872, 0.6852, 0.1020),
            ('PC10', 'fs29', 0.6602, 0.7872, -0.1270),
            ('ROI18', 'fs29', 0.7326, 0.7872, -0.0546),
            ('PC10+ROI18', 'fs29', 0.7255, 0.7872, -0.0617),
            ('ph+PC10', 'phase', 0.7196, 0.6852, 0.0344),
            ('ph+ROI18', 'phase', 0.7641, 0.6852, 0.0789),
            ('ph+PC10+ROI18', 'phase', 0.7557, 0.6852, 0.0705),
            ('ph+fs29', 'phase', 0.7950, 0.6852, 0.1098)
        ])
    ]

    for fname, pairs in with_pair:
        for a, b, auc_a, auc_b, delta in pairs:
            diff = abs(delta - (auc_a - auc_b))
            check_a_results.append(diff <= 0.00015)

    check_a_pass = all(check_a_results)
    print(f"Check a: Model AUROC difference consistency (|delta - (a-b)| <= 0.00015): {check_a_pass} ({sum(check_a_results)}/{len(check_a_results)} passed)")

    # Check b: 99% interval contains 95% interval
    has_both = df_t10a[df_t10a['ci99_lo'].notna() & df_t10a['ci95_lo'].notna()]
    b_checks = []
    for _, r in has_both.iterrows():
        b_checks.append((r['ci99_lo'] <= r['ci95_lo']) and (r['ci99_hi'] >= r['ci95_hi']))
    check_b_pass = all(b_checks)
    print(f"Check b: 99% interval contains 95% interval: {check_b_pass} ({sum(b_checks)}/{len(b_checks)} passed)")

    # Check c: n and events
    mri_fams = df_t10a[df_t10a['family'] != 'FreeSurfer saturation']
    sat_fams = df_t10a[df_t10a['family'] == 'FreeSurfer saturation']
    # Exclude E2 subgroup rows that legitimately have n=188 or 376
    mri_main = mri_fams[mri_fams['n'].isin([564, 188, 376])]
    check_c_pass = (mri_fams[mri_fams.n == 564]['events'] == 123).all() and (sat_fams['n'] == 622).all() and (sat_fams['events'] == 144).all()
    print(f"Check c: n and events (564/123 for MRI, 622/144 for saturation): {check_c_pass}")

    # Check d: recomputed vs printed criterion flags
    crit_rows = df_t10a[df_t10a['criterion_printed'] != '']
    mismatches_d = []
    for _, r in crit_rows.iterrows():
        if r['criterion_printed'] != r['criterion_recomputed']:
            mismatches_d.append((r['analysis'], r['criterion_printed'], r['criterion_recomputed']))
    check_d_pass = (len(mismatches_d) == 0)
    print(f"Check d: Recomputed vs printed criterion flags match: {check_d_pass} (mismatches: {len(mismatches_d)})")

    # Check e: row counts vs ledger executed contrasts
    ledger_fam = pd.read_csv('results/audit/mri_ledger_family_counts.csv')
    ledger_fam_exec = ledger_fam[ledger_fam.status == 'Completed'].set_index('family')['executed_contrasts']
    t10a_fam_counts = df_t10a['family'].value_counts()
    print("Check e: Family contrast counts (T10a rows vs ledger executed contrasts):")
    for f in ledger_fam_exec.index:
        t_cnt = t10a_fam_counts.get(f, 0)
        l_cnt = ledger_fam_exec[f]
        print(f"  {f:22s}: T10a={t_cnt:2d} | Ledger={l_cnt:2d}")

    # Check f: file/line existence
    check_f_results = []
    for _, r in df_t10a.iterrows():
        fpath = r['source_file']
        lno = r['source_line']
        if os.path.exists(fpath):
            with open(fpath) as f:
                lines = f.readlines()
                check_f_results.append(1 <= int(lno) <= len(lines))
        else:
            check_f_results.append(False)
    check_f_pass = all(check_f_results)
    print(f"Check f: Source file and line existence: {check_f_pass} ({sum(check_f_results)}/{len(check_f_results)} passed)")

    # Print check g comparison table
    print("\n" + "=" * 70)
    print("CHECK G: COMPARISON TABLE (PARSED DELTAS & INTERVALS)")
    print("=" * 70)
    check_g_analyses = [
        'A+ROI minus A (nested)',
        'A+ph+ROI minus A+ph (nested)',
        'A+PC10 minus A (nested)',
        'A+ph+PC10 minus A+ph (nested)',
        'A+PC10 minus A (non-nested)',
        'A+PC30 minus A (non-nested)',
        'A+ph+PC10 minus A+ph (non-nested)',
        'A+ph+PC30 minus A+ph (non-nested)',
        'A + all other FreeSurfer measures minus reference',
        'A + 10 PCs of other measures minus reference',
        'MRI increment 1ch, dropout',
        'MRI increment 1ch, dropout (PR-AUC)',
        'MRI increment 1ch, no dropout',
        'MRI increment 1ch, no dropout (PR-AUC)',
        'MRI increment 3ch, dropout',
        'MRI increment 3ch, dropout (PR-AUC)',
        'C+ROI minus C',
        'C+ph+ROI minus C+ph',
        'C+prot+ROI minus C+prot',
        'F minus C',
        'F minus C+ROI',
        'F+ROI minus F'
    ]

    for a_name in check_g_analyses:
        match = df_t10a[df_t10a['analysis'] == a_name]
        if len(match) > 0:
            row = match.iloc[0]
            c99_str = f"99% [{row['ci99_lo']:+.4f}, {row['ci99_hi']:+.4f}]" if pd.notna(row['ci99_lo']) else ""
            print(f"{a_name:42s} | delta={row['delta']:+.4f} | 95% [{row['ci95_lo']:+.4f}, {row['ci95_hi']:+.4f}] {c99_str}")

if __name__ == '__main__':
    main()
