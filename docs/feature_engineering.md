# Feature Engineering

## 1. Clinical Features (10 Features)
- Demographics: Age, Sex, Education (`PTEDUCAT`).
- Genetics: APOE4 allele count ($0, 1, 2$).
- Cognitive Assessments:
  - Mini-Mental State Examination (`MMSCORE`)
  - Clinical Dementia Rating Sum of Boxes (`CDRSB`)
  - Global Clinical Dementia Rating (`CDGLOBAL`)
  - Functional Activities Questionnaire (`FAQTOTAL`)
  - Alzheimer's Disease Assessment Scale Cognitive Subscale: 11-item (`TOTSCORE`) and 13-item (`TOTAL13`)

## 2. FreeSurfer Morphometry (29 Features)
Regional morphometric measures extracted from FreeSurfer:
- Cortical thickness (`*TA` columns): Preserved in native millimeter units.
- Subcortical and regional volumes (`*SV` columns): Normalized by Estimated Total Intracranial Volume (`ST10CV`) to control for head size:
  $$\tilde{V}_{\text{region}} = \frac{V_{\text{region}}}{\text{ST10CV}} \times 1000.0$$

## 3. Subcortical MRI ROI Features (16 Primary Features)
Extracted using the Harvard-Oxford Subcortical Structural Atlas ($2.0\text{ mm}$ MNI space):
- **Intensity Reference**: Bilateral cerebral white matter median ($N_{\text{vox}} = 62,732$).
- **Tissue Contrast**: Regional medians minus white matter median for Left/Right Hippocampus, Left/Right Amygdala, and Left/Right Lateral Ventricles.
- **Tissue Heterogeneity**: Standard deviations across voxels within each target ROI.
- **Bilateral Composites**: Bilateral Hippocampus median, Bilateral Amygdala median, Bilateral Lateral Ventricle median.
- **Ventricle-to-Hippocampus Difference**: Contrast marker for ex vacuo ventricular dilation and medial temporal lobe atrophy:
  $$\Delta_{\text{vent-hippo}} = \text{Ventricle}_{\text{bilat, med}} - \text{Hippo}_{\text{bilat, med}}$$
