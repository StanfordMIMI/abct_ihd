import pandas as pd
import os
from utils.constants import MRN_ACC_PATH, SAVE_PATH, SPLIT_PATH

def get_mrn_acc():
    '''
    Get mrn and accession data.
    '''
    if os.path.isfile(MRN_ACC_PATH):
        mrn_acc = pd.read_csv(MRN_ACC_PATH)
        return mrn_acc
    raise RuntimeError('invalid path for mrn_acc file')
    return

def get_cohort(split, mrn_acc):
    '''
    Get patient information corresponding to mrn for one cohort.
    '''
    codebook_path = os.path.join(SPLIT_PATH, str(split), 'patientCodebook.csv')
    codebook = pd.read_csv(codebook_path, sep=',')
    codebook.columns = ['patient_id', 'name', 'mrn']

    demographics_path = os.path.join(SPLIT_PATH, str(split), 'demographics.csv')
    demographics = pd.read_csv(demographics_path, sep=',', usecols=['Patient Id', 'Gender'])
    demographics.columns = ['patient_id', 'gender']

    codebook_demographics = codebook.merge(demographics, how='inner', on='patient_id')
    codebook_demographics.gender = codebook_demographics.gender.astype(str)

    cohort = codebook_demographics.merge(mrn_acc, how='inner', on='mrn')
    cohort = cohort[['patient_id', 'mrn', 'gender', 'accession', 'ct_date']]
    return cohort

def get_all_cohorts():
    '''
    Get patient information corresponding to mrn for all cohorts.
    '''
    if os.path.isfile(os.path.join(SAVE_PATH, 'all_cohorts.csv')):
        all_cohorts = pd.read_csv(os.path.join(SAVE_PATH, 'all_cohorts.csv'))
        return all_cohorts

    all_cohorts = get_cohort(0)
    for i in range(1, 11):
        cohort_i = get_cohort(i)
        all_cohorts = pd.concat([all_cohorts, cohort_i])

    print("COHORTS mrn: ", all_cohorts['mrn'].nunique())
    all_cohorts.to_csv(os.path.join(SAVE_PATH, 'all_cohorts.csv'), index=False)
    return all_cohorts

def get_codebook(split):
    """Return dataframe of patient ids for a split.
    """
    codebook_path = os.path.join(SPLIT_PATH, str(split), 'patientCodebook.csv')
    codebook = pd.read_csv(codebook_path, sep=',')
    codebook.columns = ['patient_id', 'name', 'mrn']

    return codebook

def get_demographics(split, mrn_acc):
    '''
    Get patient demographics for a split.
    '''
    path = demographics_path = os.path.join(SPLIT_PATH, str(split), 'demographics.csv')
    demographics_id = pd.read_csv(demographics_path, sep=',', usecols=['Patient Id', 'Gender', 'Date of Birth', 'Race', 'Ethnicity'])
    demographics_id.columns = ['patient_id', 'gender', 'dob', 'race', 'ethnicity']

    codebook = get_codebook(split)
    demographics = codebook.merge(demographics_id, how='inner', on='patient_id')
    demographics = demographics[['mrn', 'gender', 'dob', 'race', 'ethnicity']]

    demographics = demographics.merge(mrn_acc, how='inner', on='mrn')
    
    demographics = demographics.assign(age_at_scan=(pd.to_datetime(demographics.ct_date, format='%Y-%m-%d %H:%M:%S') \
                                            - pd.to_datetime(demographics.dob, format='%m/%d/%Y', utc=True)).apply(lambda x: (x/pd.Timedelta(days=1)/365.25)))

    demographics = demographics[['mrn', 'accession', 'ct_date', 'gender', 'age_at_scan']]
    return demographics

def get_all_demographics(mrn_acc):
    '''
    Get demographic information for each patient in all cohorts.
    '''
    if os.path.isfile(os.path.join(SAVE_PATH, 'all_demographics.csv')):
        all_cohorts_demographics = pd.read_csv(os.path.join(SAVE_PATH, 'all_demographics.csv'))
        return all_cohorts_demographics

    all_cohorts_demographics = get_demographics(0, mrn_acc)
    for i in range(1, 11):
        demographics_i = get_demographics(i, mrn_acc)
        all_cohorts_demographics = pd.concat([all_cohorts_demographics, demographics_i])

    print("DEMOGRAPHICS mrn: ", all_cohorts_demographics['mrn'].nunique())
    all_cohorts_demographics.to_csv(os.path.join(SAVE_PATH, 'all_demographics.csv'), index=False)
    return all_cohorts_demographics

def get_diagnoses(split, cohort):
    '''
    Get diagnosis information for each patient in one cohort.
    '''
    path = os.path.join(SPLIT_PATH, str(split), 'diagnoses.csv')
    diagnoses = pd.read_csv(path, sep=',')
    diagnoses.columns = ['patient_id', 'date', 'age', 'type', 'source', 'icd9_codes', \
    'icd10_codes', 'description', 'performing_provider', 'billing_provider']

    cohort_dx = cohort.merge(diagnoses, how='inner', on='patient_id')
    cohort_dx = cohort_dx.assign(
        code_type = 'icd10',
        code=cohort_dx.icd10_codes,
        time_to_ct=(pd.to_datetime(cohort_dx.ct_date, format='%Y-%m-%d %H:%M:%S') \
                    - pd.to_datetime(cohort_dx.date, format='%m/%d/%Y', utc=True)).apply(lambda x: x/pd.Timedelta(days=1)))
    cohort_dx = cohort_dx[['mrn', 'accession', 'gender', 'ct_date', 'code_type', 'code', 'time_to_ct']]
    return cohort_dx

def get_all_diagnoses(mrn_acc):
    '''
    Get diagnosis information for each patient in all cohorts.
    '''
    if os.path.isfile(os.path.join(SAVE_PATH, 'all_diagnoses.csv')):
        all_cohorts_dx = pd.read_csv(os.path.join(SAVE_PATH, 'all_diagnoses.csv'))
        return all_cohorts_dx

    all_cohorts_dx = get_diagnoses(0, get_cohort(0, mrn_acc))
    for i in range(1, 11):
        dx_i = get_diagnoses(i, get_cohort(i, mrn_acc))
        all_cohorts_dx = pd.concat([all_cohorts_dx, dx_i])

    print("DX mrn: ", all_cohorts_dx['mrn'].nunique())
    all_cohorts_dx.to_csv(os.path.join(SAVE_PATH, 'all_diagnoses.csv'), index=False)
    return all_cohorts_dx

def get_labs(split, cohort):
    '''
    Get lab information for each patient in one cohort from cohort_desired_labs.csv.
    '''
    path = os.path.join(SPLIT_PATH, str(split), 'cohort_desired_labs.csv')
    labs = pd.read_csv(path, sep=',')
    labs.columns = ['patient_id', 'mrn', 'ct_date', 'result_date', 'type', 'value']

    cohort_labs = cohort.merge(labs, how='inner', on='patient_id', suffixes=('', '_DROP')) \
                        .filter(regex='^(?!.*_DROP)')
    cohort_labs = cohort_labs.assign(
        time_to_ct=(pd.to_datetime(cohort_labs.ct_date, format='%Y-%m-%d %H:%M:%S') \
                   - pd.to_datetime(cohort_labs.result_date, format='%m/%d/%Y %H:%M', utc=True)) \
                   .apply(lambda x: x/pd.Timedelta(days=1)))

    cohort_labs = cohort_labs[['mrn', 'accession', 'gender', 'ct_date', 'type', 'value', 'time_to_ct']]
    return cohort_labs

def get_all_labs(mrn_acc):
    '''
    Get lab information for each patient in all cohorts.
    '''
    if os.path.isfile(os.path.join(SAVE_PATH, 'all_labs.csv')):
        all_cohorts_labs = pd.read_csv(os.path.join(SAVE_PATH, 'all_labs.csv'))
        return all_cohorts_labs

    all_cohorts_labs = get_labs(0, get_cohort(0, mrn_acc))
    for i in range(1, 11):
        labs_i = get_labs(i, get_cohort(i, mrn_acc))
        all_cohorts_labs = pd.concat([all_cohorts_labs, labs_i])

    print("LABS mrn: ", all_cohorts_labs['mrn'].nunique())
    all_cohorts_labs.to_csv(os.path.join(SAVE_PATH, 'all_labs.csv'), index=False)
    return all_cohorts_labs

def get_lab_tests(split, cohort):
    '''
    Get lab information for each patient in one cohort from labs.csv.
    '''
    path = os.path.join(SPLIT_PATH, str(split), 'labs.csv')
    labs = pd.read_csv(path, sep=',')
    '''
    labs = labs[['Patient Id', 'Result Date', 'Lab', 'Result', 'Value', 'Units']]
    labs.columns = ['patient_id', 'result_date', 'type', 'result', 'value', 'units']
    '''
    labs.columns = ['patient_id', 'order_date', 'taken_date', 'result_date', 'age', \
                    'lab', 'result', 'value', 'reference_low', 'reference_high', \
                    'units', 'abnormal', 'comment', 'authorizing_provider']

    # remove rows with missing lab values
    labs = labs[labs.value.str.lower() != 'n/a']
    labs = labs.dropna(subset=['value', 'lab', 'result'])

    # abnormally high: 1, normal: 0, abnormally low: -1
    '''
    labs.loc[labs['abnormal'] == 'Abnormally High', ['abnormal']] = 1
    labs.loc[labs['abnormal'] == 'Abnormally Low', ['abnormal']] = -1
    labs = labs.fillna(0)
    '''

    cohort = cohort[['patient_id', 'mrn']].drop_duplicates()

    cohort_labs = cohort.merge(labs, how='inner', on='patient_id', suffixes=('', '_DROP')) \
                        .filter(regex='^(?!.*_DROP)')
    cohort_labs = cohort_labs.drop(columns=['patient_id'])

    '''
    cohort_labs['valid_result_date'] = pd.to_datetime(cohort_labs.result_date, format='%m/%d/%Y %H:%M', utc=True, errors='coerce')
    cohort_labs = cohort_labs.dropna(subset=['valid_result_date'])
    cohort_labs = cohort_labs.assign(
        time_to_ct=(pd.to_datetime(cohort_labs.ct_date, format='%Y-%m-%d', utc=True) \
                   - pd.to_datetime(cohort_labs.result_date, format='%m/%d/%Y %H:%M', utc=True)) \
                   .apply(lambda x: x.days))
    cohort_labs = cohort_labs[['mrn', 'accession', 'type', 'result', 'value', 'units', 'time_to_ct']]
    '''

    return cohort_labs

def get_all_lab_tests():
    if os.path.isfile(os.path.join(SAVE_PATH, 'lab_tests_combined.csv')):
        all_cohorts_lab_tests = pd.read_csv(os.path.join(SAVE_PATH, 'lab_tests_combined.csv'))
        return all_cohorts_lab_tests

    all_cohorts_lab_tests = get_lab_tests(0, get_cohort(0))
    for i in range(1, 11):
        print(i)
        lab_tests_i = get_lab_tests(i, get_cohort(i))
        all_cohorts_lab_tests = pd.concat([all_cohorts_lab_tests, lab_tests_i])

    print("LAB TESTS mrn: ", all_cohorts_lab_tests['mrn'].nunique())
    all_cohorts_lab_tests.to_csv(os.path.join(SAVE_PATH, 'lab_tests_combined.csv'), index=False)
    return all_cohorts_lab_tests

def get_procedures(split, cohort):
    '''
    Get procedures for each patient in one cohort.
    '''
    path = os.path.join(SPLIT_PATH, str(split), 'procedures.csv')
    procedures = pd.read_csv(path, sep=',')
    procedures.columns = ['patient_id', 'date', 'age', 'code', 'code_type', 'description', \
    'performing_provider', 'billing_provider']
    procedures = procedures[procedures['code_type'] == 'CPT']

    cohort_proc = cohort.merge(procedures, how='inner', on='patient_id')
    cohort_proc = cohort_proc.assign(
        code_type = 'cpt',
        time_to_ct=(pd.to_datetime(cohort_proc.ct_date, format='%Y-%m-%d %H:%M:%S') \
                    - pd.to_datetime(cohort_proc.date, format='%m/%d/%Y', utc=True)).apply(lambda x: x/pd.Timedelta(days=1)))
    cohort_proc = cohort_proc[['mrn', 'accession', 'ct_date', 'code_type', 'code', 'time_to_ct']]
    return cohort_proc

def get_all_procedures(mrn_acc):
    '''
    Get procedures for each patient in all cohorts.
    '''
    if os.path.isfile(os.path.join(SAVE_PATH, 'all_procedures.csv')):
        all_cohorts_proc = pd.read_csv(os.path.join(SAVE_PATH, 'all_procedures.csv'))
        return all_cohorts_proc

    all_cohorts_proc = get_procedures(0, get_cohort(0, mrn_acc))
    for i in range(1, 11):
        proc_i = get_procedures(i, get_cohort(i, mrn_acc))
        all_cohorts_proc = pd.concat([all_cohorts_proc, proc_i])

    print("PROC mrn: ", all_cohorts_proc['mrn'].nunique())
    all_cohorts_proc.to_csv(os.path.join(SAVE_PATH, 'all_procedures.csv'), index=False)
    return all_cohorts_proc

def get_vitals(split, cohort):
    '''
    Get vitals information for each patient in one cohort.
    '''
    path = os.path.join(SPLIT_PATH, str(split), 'cohort_vitals.csv')
    vitals = pd.read_csv(path, sep=',').drop(columns=['id'])
    vitals.columns = ['ct_date', 'mrn', 'patient_id', 'date', 'type', 'value']
    cohort_vitals = cohort.merge(vitals, how='inner', on='patient_id', suffixes=('', '_DROP')) \
                          .filter(regex='^(?!.*_DROP)')
    cohort_vitals = cohort_vitals.assign(
        time_to_ct=(pd.to_datetime(cohort_vitals.ct_date, format='%Y-%m-%d %H:%M:%S') \
                   - pd.to_datetime(cohort_vitals.date, format='%Y-%m-%d', utc=True)).apply(lambda x: x/pd.Timedelta(days=1)))

    cohort_vitals = cohort_vitals[['mrn', 'accession', 'gender', 'ct_date', 'type', 'value', 'time_to_ct']]
    return cohort_vitals

def get_all_vitals(mrn_acc):
    '''
    Get vitals information for each patient in all cohorts.
    '''
    if os.path.isfile(os.path.join(SAVE_PATH, 'all_vitals.csv')):
        all_cohorts_vitals = pd.read_csv(os.path.join(SAVE_PATH, 'all_vitals.csv'))
        return all_cohorts_vitals

    all_cohorts_vitals = get_vitals(0, get_cohort(0, mrn_acc))
    for i in range(1, 11):
        vitals_i = get_vitals(i, get_cohort(i, mrn_acc))
        all_cohorts_vitals = pd.concat([all_cohorts_vitals, vitals_i])

    print("VITALS mrn: ", all_cohorts_vitals['mrn'].nunique())
    all_cohorts_vitals.to_csv(os.path.join(SAVE_PATH, 'all_vitals.csv'), index=False)
    return all_cohorts_vitals

def get_drugs(split, cohort):
    '''
    Get drugs information for each patient in one cohort from med_orders.csv.
    -RXCUI refers to RxNorm CUI for each drug
    '''
    path = os.path.join(SPLIT_PATH, str(split), 'med_orders.csv')
    drugs = pd.read_csv(path, sep=',')
    drugs.columns = ['patient_id', 'mrn', 'ct_date', 'start_date', 'RXCUI', 'order_status']
    drugs = drugs[(drugs['order_status'] == 'Sent') | (drugs['order_status'] == 'Completed') | (drugs['order_status'] == 'Dispensed') | (drugs['order_status'] == 'Verified')]
    cohort_drugs = cohort.merge(drugs, how='inner', on='patient_id', suffixes=('', '_DROP')) \
                        .filter(regex='^(?!.*_DROP)')
    cohort_drugs = cohort_drugs.assign(
        time_to_ct=(pd.to_datetime(cohort_drugs.ct_date, format='%Y-%m-%d %H:%M:%S') \
                   - pd.to_datetime(cohort_drugs.start_date, format='%m/%d/%Y %H:%M', utc=True)) \
                   .apply(lambda x: x/pd.Timedelta(days=1)))

    cohort_drugs = cohort_drugs[['mrn', 'accession', 'ct_date', 'RXCUI', 'time_to_ct']]
    return cohort_drugs

def get_all_drugs(mrn_acc):
    '''
    Get drugs (prescription) information for each patient in all cohorts.
    '''
    if os.path.isfile(os.path.join(SAVE_PATH, 'all_drugs.csv')):
        all_cohorts_drugs = pd.read_csv(os.path.join(SAVE_PATH, 'all_drugs.csv'))
        return all_cohorts_drugs

    all_cohorts_drugs = get_drugs(0, get_cohort(0, mrn_acc))
    for i in range(1, 11):
        drugs_i = get_drugs(i, get_cohort(i, mrn_acc))
        all_cohorts_drugs = pd.concat([all_cohorts_drugs, drugs_i])

    print("DRUGS mrn: ", all_cohorts_drugs['mrn'].nunique())
    all_cohorts_drugs.to_csv(os.path.join(SAVE_PATH, 'all_drugs.csv'), index=False)
    return all_cohorts_drugs