# Features Generation
These files can be used to extract features from the dataset. 

## Files
To create a feature matrix from demographic information, labs and vitals, and ICD-10
and CPT codes, run the following script:
- `cohort_features.py`

## Description of Features
Demographic information, lab and vital measurements, diagnosis codes, and procedure codes are arranged in a matrix where each row represented a patient. Only data collected prior to the date of the CT scan are used.  

To construct features for each patient, gender, is constructed as a binary variable (1=Male, 0=Female), age at the time of the CT scan is presented in years as a continous variable. The body mass index (BMI) is obtained as the most recent available measurement. 

Diagnosis, procedure codes, drugs and vitals are restricted to 0-1 years prior to time of acquisition of the CT scan.

ICD-10 diagnosis codes, CPT procedure codes, and ATC drug codes are encoded with count variables.  To reduce the number of ICD-10 and CPT code categories, they are grouped based on their underlying ontology (see `utils/*_hierarchy_ranges.csv`).  Drug codes are mapped from RxNorm RXCUI codes to ATC drug codes, keeping only the first two digits of ATC codes (ATC2).

Lab and vital measurements are aggregated with an exponentially weighted average based on the time between the date of
the measurement and the date of the CT scan, with higher weights assigned to more recent results.

## Description of raw data
The original data is obtained in 11 subsets of patients (called "splits") which have varying number of patients but all share the same file naming convention. Since the final cohort represents a subset of the original (after applying exclusion criteria), these files are cached to contain only relevant measurements for patients in the analysis cohort (`all_*.csv` files in cache).
