# Dataset & Cohort Design

## Data Source
The multi-modal data analyzed in this investigation originated from the Alzheimer's Disease Neuroimaging Initiative repository (`adni.loni.usc.edu`). ADNI research data access is administered under formal data use agreements requiring registered institutional credentialing.

## Cohort Flow and Eligibility Criteria
Starting from 1,199 screened study participants:
1. **Diagnosis**: Baseline diagnosis of Mild Cognitive Impairment (Early MCI or Late MCI).
2. **Age Range**: 55 to 90 years at baseline.
3. **Follow-Up Horizon**: Followed for at least 730 days (24 months) or observed conversion to Alzheimer's disease dementia prior to 730 days.
4. **Exclusions**: Reversions to cognitively normal status, progression to non-AD dementias (e.g., Lewy body, vascular), and missing baseline cognitive scores.

### Final Cohort Partitioning
- **Primary Eligible Development Cohort**: $N = 622$ participants (144 converters, 478 non-converters; conversion rate: $23.15\%$).
- **Matched Development Cohort with Usable Baseline 3D MRI**: $N = 564$ participants (123 converters, 441 non-converters; conversion rate: $21.81\%$).
- **Locked Internal Holdout**: $N = 156$ participants (strictly partitioned and preserved for final validation; 0 holdout cases entered development experiments).

## Multi-Phase Distribution ($N = 564$ Matched Cohort)
- **ADNI1**: $N = 188$ (72 converters, 116 non-converters; $38.3\%$ conversion rate)
- **ADNI2**: $N = 204$ (38 converters, 166 non-converters; $18.6\%$ conversion rate)
- **ADNI3**: $N = 93$ (10 converters, 83 non-converters; $10.8\%$ conversion rate)
- **ADNIGO**: $N = 79$ (3 converters, 76 non-converters; $3.8\%$ conversion rate)
