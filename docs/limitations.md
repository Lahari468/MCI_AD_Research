# Limitations & Scientific Boundaries

## 1. Spatial Registration Resolution
- MRI preprocessing utilized linear affine registration (FSL FLIRT 12 DOF) to MNI152 2 mm isotropic space.
- While global alignment is verified, non-linear deformable registration (e.g., FNIRT or ANTs SyN) was not applied to the volumes. Consequently, standard-space subcortical atlas masks partially overlap surrounding CSF or white matter at regional boundaries due to unmodeled subject-specific anatomical variations.

## 2. Multi-Site Scanner & Acquisition Heterogeneity
- The ADNI dataset spans multiple phases (ADNI1, ADNI2, ADNI3, ADNIGO) across more than 50 clinical sites with 1.5T and 3.0T field strengths.
- While N4 bias correction and brain-masked z-scoring reduce field inhomogeneity, T1 intensity contrast depends non-linearly on scanner hardware, pulse sequences, and flip angles.
- Bilateral white matter median subtraction provides an intra-subject baseline shift, but cannot eliminate inter-scanner contrast variability.

## 3. Statistical Power in Small ADNI Phases
- In ADNI3 ($N_{\text{pos}} = 10$) and ADNIGO ($N_{\text{pos}} = 3$), converter sample sizes are small.
- Phase-stratified AUROC estimates in these subgroups carry wide error margins and should not be interpreted as evidence of scanner-dependent incremental value.

## 4. Scientific Scope & Negative Findings
- The empirical finding of non-significance ($\Delta \text{AUROC} = +0.0044$, $p = 0.2440$) demonstrates that subcortical MRI features do not add statistically significant predictive value beyond Clinical + FreeSurfer for 24-month progression in this cohort.
- This finding must **not** be misconstrued as claiming that structural MRI has "no biological value" or that FreeSurfer "completely captures" all possible MRI information. Structural MRI remains clinically vital for differential diagnosis, exclusion of secondary etiologies, and long-term disease monitoring.
