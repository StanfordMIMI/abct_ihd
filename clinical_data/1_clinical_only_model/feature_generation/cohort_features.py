import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
import cohort_cache 
from utils.constants import DATA_HOME, SPLIT_PATH, SAVE_PATH

def exp_weighted_avg(group):  
    
    vals = np.array(group.value)
    time = np.array(group.time_to_ct)
        
    weights = np.exp(-np.abs(time)-1)

    dot = np.dot(vals, weights) / weights.sum()

    return np.NaN if np.isnan(dot) else dot

def features_demographics(cohort):
    """Return dataframe of all patient demographics with one hot encoding of features.
    """
    all_demographics = cohort_cache.get_all_demographics(cohort)

    # Filter cohort.
    all_demographics = all_demographics.merge(cohort, how='right', on=['mrn','accession','ct_date'])
    all_demographics = all_demographics[['mrn', 'accession', 'ct_date', 'gender', 'age_at_scan']]

    # Create categorical features
    all_demographics.replace({'gender': {'Male': 1, 'Female': 0}}, inplace=True)
    
    return all_demographics

def features_labs_raw(cohort):
    """Return dataframe of all patient labs without any encoding.
    """
    all_labs = cohort_cache.get_all_labs(cohort)

    # Filter cohort
    all_labs = all_labs.merge(cohort, how='inner', on=['mrn','accession'])

    # Only keep measurements prior to CT scan
    all_labs = all_labs[(all_labs['time_to_ct'] >= 0) & all_labs['time_to_ct'] <= 365.25]
    all_labs = all_labs[['mrn', 'accession', 'time_to_ct', 'type', 'value']]
    
    return all_labs

def features_labs(cohort):
    """Return dataframe of all patient labs with ordinal encoding for features.
    Labs used: ['gluc', 'chol_total', 'trig', 'hba1c', 'chol_hdl', 'chol_ldl']
    """
    all_labs_raw = features_labs_raw(cohort)
    desired_labs = {'chol_ldl', 'trig', 'chol_hdl', 'hba1c', 'gluc', 'chol_total'}
    all_labs_raw = all_labs_raw[all_labs_raw['type'].astype(str).isin(desired_labs)]
    all_labs = all_labs_raw.loc[:,['mrn', 'accession', 'time_to_ct', 'type', 'value']]
    
    # Organize measurements by type and aggregate for each mrn/accession pair
    avg_labs = all_labs.sort_values(['time_to_ct'], ascending=True) \
                       .groupby(['mrn', 'accession', 'type']) \
                       .apply(exp_weighted_avg) \
                       .to_frame() \
                       .reset_index()
    avg_labs.columns = ['mrn', 'accession', 'type', 'exp_wt_value']
    avg_labs = avg_labs.pivot_table('exp_wt_value', ['mrn', 'accession'], 'type') \
                       .reset_index()
    new_col_names = [(i, 'exp_wt_'+i) for i in avg_labs.columns.values if i not in ['mrn','accession']]
    avg_labs.rename(columns=dict(new_col_names), inplace=True)
    
    # Summarize measurements by # of times measured
    n_labs = all_labs.sort_values(['time_to_ct'], ascending=True) \
                            .groupby(['mrn', 'accession', 'type']) \
                            .count() \
                            .drop('time_to_ct', axis=1) \
                            .reset_index()
    
    n_labs.columns = ['mrn','accession','type', 'n_value']
    n_labs = n_labs.pivot_table('n_value', ['mrn', 'accession'], 'type') \
                           .reset_index()
    new_col_names = [(i, 'num_times_measured_'+i) for i in n_labs.columns.values if i not in ['mrn','accession']]
    n_labs.rename(columns=dict(new_col_names), inplace=True)
    n_labs.fillna(0, inplace=True) #only for counts

    # Organize measurements by type and keep most recent for each mrn/accession pair
    latest_labs = all_labs.sort_values(['time_to_ct'], ascending=True) \
                            .drop_duplicates(['mrn', 'accession', 'type'])\
                            .drop('time_to_ct', axis=1)
    
    latest_labs.columns = ['mrn','accession','type', 'latest_value']
    latest_labs = latest_labs.pivot_table('latest_value', ['mrn', 'accession'], 'type') \
                           .reset_index()
    new_col_names = [(i, 'latest_value_'+i) for i in latest_labs.columns.values if i not in ['mrn','accession']]
    latest_labs.rename(columns=dict(new_col_names), inplace=True)

    # Get entire cohort
    all_labs = avg_labs.merge(latest_labs, how='outer', on=['mrn','accession'])\
                            .merge(n_labs, how='outer', on=['mrn','accession'])\
                            .merge(cohort[['mrn', 'accession']], how='right', on=['mrn','accession'])

    return all_labs

def features_vitals_raw(cohort):
    """Return dataframe of all patient vitals without any encoding.
    """
    all_vitals = cohort_cache.get_all_vitals(cohort)

    # Filter cohort
    all_vitals = all_vitals.merge(cohort, how='inner', on=['mrn','accession'])

    # Only keep measurements prior to CT scan
    all_vitals = all_vitals[(all_vitals['time_to_ct'] >= 0) & all_vitals['time_to_ct'] <= 365.25]
    all_vitals = all_vitals[['mrn', 'accession', 'time_to_ct', 'type', 'value']]

    return all_vitals

def features_vitals(cohort):
    """Return dataframe of all patient vitals with ordinal encoding of features.
    Vitals available: ['height_m' 'weight_kg' 'bmi_calc' 'BMI' 'systolic' 'diastolic']
    Vitals used: ['bmi_calc' 'systolic' 'diastolic']
    """
    desired_types = ['bmi_calc', 'systolic', 'diastolic']
    
    all_vitals_raw = features_vitals_raw(cohort)
    
    all_vitals_raw = all_vitals_raw[all_vitals_raw['type'].astype(str).isin(desired_types)]
    all_vitals = all_vitals_raw[['mrn', 'accession', 'time_to_ct', 'type', 'value']]
    
    # Organize measurements by type and aggregate for each mrn/accession pair
    avg_vitals = all_vitals.sort_values(['time_to_ct'], ascending=True) \
                           .groupby(['mrn', 'accession', 'type']) \
                           .apply(exp_weighted_avg) \
                           .to_frame() \
                           .reset_index()
    
    avg_vitals.columns = ['mrn','accession','type', 'exp_wt_value']
    avg_vitals = avg_vitals.pivot_table('exp_wt_value', ['mrn', 'accession'], 'type') \
                           .reset_index()
    new_col_names = [(i, 'exp_wt_lab_'+i) for i in avg_vitals.columns.values if i not in ['mrn','accession']]
    avg_vitals.rename(columns=dict(new_col_names), inplace=True)
    # Summarize measurements by # of times measured
    n_vitals = all_vitals.sort_values(['time_to_ct'], ascending=True) \
                            .groupby(['mrn', 'accession', 'type']) \
                            .count() \
                            .drop('time_to_ct', axis=1) \
                            .reset_index()
    
    n_vitals.columns = ['mrn','accession','type', 'n_value']
    n_vitals = n_vitals.pivot_table('n_value', ['mrn', 'accession'], 'type') \
                           .reset_index()
    new_col_names = [(i, 'num_times_measured_'+i) for i in n_vitals.columns.values if i not in ['mrn','accession']]
    n_vitals.rename(columns=dict(new_col_names), inplace=True)
    n_vitals.fillna(0, inplace=True) #only for counts
    # Organize measurements by type and keep most recent for each mrn/accession pair
    latest_vitals = all_vitals.sort_values(['time_to_ct'], ascending=True) \
                            .drop_duplicates(['mrn', 'accession', 'type'])\
                            .drop('time_to_ct', axis=1)
    
    latest_vitals.columns = ['mrn','accession','type', 'latest_value']
    latest_vitals = latest_vitals.pivot_table('latest_value', ['mrn', 'accession'], 'type') \
                           .reset_index()
    new_col_names = [(i, 'latest_value_'+i) for i in latest_vitals.columns.values if i not in ['mrn','accession']]
    latest_vitals.rename(columns=dict(new_col_names), inplace=True)
    # Get entire cohort
    all_vitals = avg_vitals.merge(latest_vitals, how='outer', on=['mrn','accession'])\
                            .merge(n_vitals, how='outer', on=['mrn','accession'])\
                            .merge(cohort[['mrn', 'accession']], how='right', on=['mrn','accession'])

    #keep only most recent bmi value:
    all_vitals.rename(columns={'latest_value_bmi_calc':'bmi'}, inplace=True)
    all_vitals.drop([x for x in all_vitals.columns if 'bmi_calc' in x], axis=1, inplace=True)

    return all_vitals

def features_diagnoses(cohort):
    """Return dataframe of all patient ICD10 codes.
    """
    def read_icd10_dict(file_name):
        icd10_mapping = pd.read_csv(file_name)
        return icd10_mapping[icd10_mapping['HIERARCHY']==2]

    def map_diagnoses(all_diagnoses):
        icd10_mapping = read_icd10_dict('./utils/ICD10_hierarchy_ranges.csv')

        for index, diagnosis in all_diagnoses.iterrows():
            mapped = False
            for _, row in icd10_mapping.iterrows():
                if diagnosis['code'] >= row['RNG_MIN'] and diagnosis['code'] <= row['RNG_MAX']:
                    all_diagnoses.at[index, 'code'] = row['UNIQUE_NAME']
                    mapped = True
                    break
            if not mapped:
                all_diagnoses.at[index, 'code'] = np.nan

        all_diagnoses = all_diagnoses.dropna(subset=['code'])
        return all_diagnoses

    all_diagnoses = cohort_cache.get_all_diagnoses(cohort)
    
    # Filter cohort
    all_diagnoses = all_diagnoses.merge(cohort, how='inner', on=['mrn','accession'])
    all_diagnoses = all_diagnoses[(all_diagnoses['time_to_ct'] >= 0) & (all_diagnoses['time_to_ct'] <= 365.25)]
    all_diagnoses = all_diagnoses[['mrn', 'accession', 'code']]
    all_diagnoses = all_diagnoses.dropna()
    
    # Split comma separated codes (i.e. those where 2+ codes are in one field)
    all_diagnoses['code'] = all_diagnoses['code'].str.split(',')
    all_diagnoses = all_diagnoses.explode('code').reset_index(drop=True)

    # Keep only ICD10 category
    all_diagnoses['code'] = all_diagnoses['code'].map(lambda x: x.split('.', 1)[0].strip())
    
    # Map to IDC10 chapter
    all_diagnoses = map_diagnoses(all_diagnoses)
    
    # encoding
    all_diagnoses = all_diagnoses.pivot_table(index=['mrn','accession'], columns='code', aggfunc='size', fill_value=0)\
                                    .reset_index()
    
    # Get entire cohort
    all_diagnoses = all_diagnoses.merge(cohort[['mrn', 'accession']], how='right', on=['mrn', 'accession'])
    
    return all_diagnoses.fillna(value=0)

def features_procedures(cohort):
    """Return dataframe of all patient CPT codes.
    """
    def read_cpt_dict(file_name):
        cpt_mapping = pd.read_csv(file_name)
        cpt_mapping = cpt_mapping[cpt_mapping['HIERARCHY']=='H2']
        cpt_mapping['range'] = cpt_mapping.apply(lambda x : list(range(int(x['RNG_MIN']), int(x['RNG_MAX'])+1)), 1)
        cpt_mapping = cpt_mapping[['range', 'UNIQUE_NAME']].explode('range')
        cpt_mapping_dict = pd.Series(cpt_mapping['UNIQUE_NAME'].values,index=cpt_mapping.range).to_dict()
        return cpt_mapping_dict

    def map_numeric(all_procedures_num):
        cpt_mapping_dict = read_cpt_dict('./utils/CPT_hierarchy_ranges.csv')
        all_procedures_num = all_procedures_num.assign(mapping = \
            all_procedures_num['code'].astype(int).map(cpt_mapping_dict) \
                                      .fillna(np.nan))
        all_procedures_num = all_procedures_num.dropna(subset=['mapping'])
        return all_procedures_num

    all_procedures = cohort_cache.get_all_procedures(cohort)

    # Filter cohort
    all_procedures = all_procedures.merge(cohort, how='inner', on=['mrn','accession'])
    all_procedures = all_procedures[(all_procedures['time_to_ct'] >= 0) & (all_procedures['time_to_ct'] <= 365.25)]
    all_procedures = all_procedures[['mrn', 'accession', 'code']]
    all_procedures = all_procedures.dropna()

    # Map CPT codes to categories
    all_procedures_num = all_procedures[~all_procedures['code'].str.contains('[A-Za-z, ]')]
    all_procedures = map_numeric(all_procedures_num)

    # encoding
    all_procedures = all_procedures.pivot_table(index=['mrn','accession'], columns='mapping', aggfunc='size', fill_value=0)\
                                    .reset_index()

    # Get entire cohort
    all_procedures = all_procedures.merge(cohort[['mrn', 'accession']], how='right', on=['mrn', 'accession'])

    return all_procedures.fillna(value=0)

def features_drugs(cohort):
    """
    Return dataframe of all patient ATC codes.
    """
    def read_atc_dict(file_name='./utils/rxcui_to_atc2.csv'):
        """
        return dictionary mapping ATC2 codes for each drug RXCUI
        """
        atc_mapping = pd.read_csv(file_name)
        atc_mapping_dict = atc_mapping.set_index('RXCUI').to_dict()['ATC2']
        return atc_mapping_dict

    def map_numeric(all_drugs):
        atc_mapping_dict = read_atc_dict()
        all_drugs = all_drugs.assign(mapping = \
            all_drugs['RXCUI'].astype(str).map(atc_mapping_dict) \
                                      .fillna(np.nan))
        all_drugs = all_drugs.dropna(subset=['mapping'])
        return all_drugs

    all_drugs = cohort_cache.get_all_drugs(cohort)

    # Filter cohort
    all_drugs = all_drugs.merge(cohort, how='inner', on=['mrn','accession'])
    all_drugs = all_drugs[(all_drugs['time_to_ct'] >= 0) & (all_drugs['time_to_ct'] <= 365.25)]
    all_drugs = all_drugs[['mrn', 'accession', 'code']]
    all_drugs = all_drugs.dropna()

    # Map RXCUI codes to ATC2 categories
    all_drugs = map_numeric(all_drugs)

    # encoding
    all_drugs = all_drugs.pivot_table(index=['mrn','accession'], columns='mapping', aggfunc='size', fill_value=0)\
                                    .reset_index()

    # Get entire cohort
    all_drugs = all_drugs.merge(cohort[['mrn', 'accession']], how='right', on=['mrn', 'accession'])

    return all_drugs.fillna(value=0)

def get_feature_matrix_raw(cohort, cohort_name):
    """
    Inputs:
        -cohort: pandas DataFrame containing at least 3 columns: mrn, accession, ct_date
        -cohort_name: string containing prefix for output file
    Returns:
        -feature_matrix: pandas DataFrame containing all features for each patient
    """
    all_features = None
    file_name = cohort_name + '_cont_features.csv'
    if os.path.isfile(os.path.join(SAVE_PATH, file_name)):
        all_features = pd.read_csv(os.path.join(SAVE_PATH, file_name))
    else:
        all_demographics = features_demographics(cohort)
        all_vitals = features_vitals(cohort)
        all_diagnoses = features_diagnoses(cohort)
        all_procedures = features_procedures(cohort)
        all_labs = features_labs(cohort) 
        all_drugs = features_drugs(cohort)

        all_features = all_demographics.merge(all_vitals, how='outer', on=['mrn', 'accession'])\
                                .merge(all_diagnoses, how='outer', on=['mrn', 'accession'])\
                                .merge(all_procedures, how='outer', on=['mrn', 'accession'])\
                                .merge(all_labs, how='outer', on=['mrn', 'accession'])\
                                .merge(all_drugs, how='outer', on=['mrn', 'accession'])\
                                .merge(cohort, how='inner', on=['mrn', 'accession'])

        all_features.to_csv(os.path.join(SAVE_PATH, file_name), index=False)

    return all_features

if __name__ == '__main__':
    cohort = cohort_cache.get_mrn_acc()

    fts = get_feature_matrix_raw(cohort=cohort, cohort_name='IHD_8139_py')