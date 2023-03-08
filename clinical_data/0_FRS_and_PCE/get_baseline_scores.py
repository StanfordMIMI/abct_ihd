"""
Implements Framingham Risk Score (FRS) and PCE (Pooled Cohort Equations) 
risk assessment helper functions
"""

from numpy import log, exp, dot, array

def get_frs(age: float, has_ht_treatment:bool, has_diabetes:bool, \
    total_chol:int, hdl_chol:int, is_male:bool, is_smoker:bool, systolic_bp:int):
    """
    returns Framingham Risk Score for a given individual given covariates

    params:
        - age: age of patient in years
        - has_ht_treatment: whether the patient is treated for HT
        - has_diabetes: whether the patient has diabetes
        - total_chol: total cholesterol in mg/dL
        - hdl_chol: HDL cholesterol in mg/dL
        - is_male: whether the patient's sex is male (as opposed to female)
        - is_smoker: whether the patient is a current smoker
        - systolic_bp: systolic blood pressure in mmHg

    coefficients obtained from https://www.ahajournals.org/doi/full/10.1161/circulationaha.107.699579, table 2
    """
    betas = {'female':{'s0':0.95012,
                    'log_age':2.32888, 
                    'log_total_chol':1.20904,
                    'log_hdl_col':-0.70833,
                    'log_sbp_treated':2.82263,
                    'log_sbp_untreated':2.76157,
                    'smoker':0.52873,
                    'diabetes':0.69154},
            'male':{'s0':0.88936,
                    'log_age':3.06117, 
                    'log_total_chol':1.12370,
                    'log_hdl_col':-0.93263,
                    'log_sbp_treated':1.99881,
                    'log_sbp_untreated':1.93303,
                    'smoker':0.65451,
                    'diabetes':0.57367}
            }
    sum2 = {'female': dot(array([2.32888, 1.20904, -.70833, 2.76157, 2.82263, .52873, .69154]), 
                                array([3.8686, 5.3504, 4.0176, 4.2400, 0.5826, 0.3423, 0.0376])),
                'male': dot(array([3.06117, 1.12370, -.93263, 1.93303, 1.99881, .65451, .57367]),
                            array([3.8560, 5.3420, 3.7686, 4.3544, .5019, .3522, .0650]))}
    if is_male:
        sex = 'male'
    else:
        sex = 'female'
    if has_ht_treatment:
        log_sbp_var = 'log_sbp_treated'
    else:
        log_sbp_var = 'log_sbp_untreated'
    sum1 = dot(array([betas[sex][var] for var in ['log_age', 'log_total_chol', 'log_hdl_col', log_sbp_var, 'smoker', 'diabetes']]),
                array([log(age), log(total_chol), log(hdl_chol), log(systolic_bp), is_smoker, has_diabetes]))
    risk = 1-betas[sex]['s0']**exp(sum1-sum2[sex])
    return risk

def get_pce(age: float, has_ht_treatment:bool, has_diabetes:bool, \
    total_chol:int, hdl_chol:int, is_male:bool, is_smoker:bool, systolic_bp:int,
    is_african_american:bool):
    """
    returns Pooled Cohort Equations risk for a given individual given covariates

    params:
        - age: age of patient in years
        - has_ht_treatment: whether the patient is treated for HT
        - has_diabetes: whether the patient has diabetes
        - total_chol: total cholesterol in mg/dL
        - hdl_chol: HDL cholesterol in mg/dL
        - is_male: whether the patient's sex is male (as opposed to female)
        - is_smoker: whether the patient is a current smoker
        - systolic_bp: systolic blood pressure in mmHg
        - is_african_american: whether the patient is African American (all others default to coefficients for White)

    coefficients obtained from https://www.ahajournals.org/doi/10.1161/01.cir.0000437741.48606.98, appendix 7, table 4
    """
    betas = {'white':{'female':{'s0':0.9665,
                                'log_age':-29.799,
                                'log_age_sq':4.884, 
                                'log_total_chol':13.540,
                                'log_age_x_total_chol':-3.114,
                                'log_hdl_chol':-13.578,
                                'log_age_x_hdl_chol':3.149,
                                'log_sbp_treated':2.019,
                                'log_age_x_sbp_treated':0,
                                'log_sbp_untreated':1.957,
                                'log_age_x_sbp_untreated':0,
                                'smoker':7.574,
                                'log_age_x_smoker':-1.665,
                                'diabetes':0.661},
                    'male':{'s0':0.91436,
                                'log_age':12.344,
                                'log_age_sq':0, 
                                'log_total_chol':11.853,
                                'log_age_x_total_chol':-2.664,
                                'log_hdl_chol':-7.990,
                                'log_age_x_hdl_chol':1.769,
                                'log_sbp_treated':1.797,
                                'log_age_x_sbp_treated':0,
                                'log_sbp_untreated':1.764,
                                'log_age_x_sbp_untreated':0,
                                'smoker':7.837,
                                'log_age_x_smoker':-1.795,
                                'diabetes':0.658}
            },
            'african_american':{'female':{'s0':0.9533,
                                    'log_age':17.114,
                                    'log_age_sq':0, 
                                    'log_total_chol':0.940,
                                    'log_age_x_total_chol':0,
                                    'log_hdl_chol':-18.920,
                                    'log_age_x_hdl_chol':4.475,
                                    'log_sbp_treated':29.291,
                                    'log_age_x_sbp_treated':-6.432,
                                    'log_sbp_untreated':27.820,
                                    'log_age_x_sbp_untreated':-6.087,
                                    'smoker':0.691,
                                    'log_age_x_smoker':0,
                                    'diabetes':0.874},
                                'male':{'s0':0.8954,
                                    'log_age':2.469,
                                    'log_age_sq':0, 
                                    'log_total_chol':0.302,
                                    'log_age_x_total_chol':0,
                                    'log_hdl_chol':-0.307,
                                    'log_age_x_hdl_chol':0,
                                    'log_sbp_treated':1.916,
                                    'log_age_x_sbp_treated':0,
                                    'log_sbp_untreated':1.809,
                                    'log_age_x_sbp_untreated':0,
                                    'smoker':0.549,
                                    'log_age_x_smoker':0,
                                    'diabetes':0.645}
            }}

    sum2 = {'white':{'female':-29.1817,'male':61.1816},
            'african_american':{'female':86.6081,'male':19.5425}}
    if is_african_american:
        race = 'african_american'
    else:
        race = 'white'
    if has_ht_treatment:
        log_sbp_var = ('log_sbp_treated','log_age_x_sbp_treated')
    else:
        log_sbp_var = ('log_sbp_untreated','log_age_x_sbp_untreated')
    if is_male:
        sex = 'male'
    else:
        sex = 'female'

    sum1 = dot(array([betas[race][sex][var] for var in ['log_age', 'log_age_sq', 'log_total_chol','log_age_x_total_chol','log_hdl_chol',\
                                                        'log_age_x_hdl_chol', log_sbp_var[0],log_sbp_var[1], 'smoker', 'log_age_x_smoker','diabetes']]),
                array([log(age), log(age)**2, log(total_chol), log(age)*log(total_chol), log(hdl_chol), \
                    log(age)*log(hdl_chol), log(systolic_bp), log(systolic_bp)*log(age), is_smoker, is_smoker*log(age),has_diabetes]))
    pce_risk = 1-betas[race][sex]['s0']**exp(sum1-sum2[race][sex])

    return pce_risk
def main():
    #PRS debug
    sample_case1 = get_frs(age=61, has_ht_treatment=False, has_diabetes=False, total_chol=180, hdl_chol=47, is_smoker=True, systolic_bp=124, is_male=False)
    sample_case2 = get_frs(age=53, is_male=True, has_ht_treatment=True, has_diabetes=True, total_chol=161, hdl_chol=55, systolic_bp=125, is_smoker=False)

    #PCE debug
    sample_case1 = get_pce(age=55, is_male=False, is_african_american=False, has_ht_treatment=False, has_diabetes=False, total_chol=213, hdl_chol=50, systolic_bp=120, is_smoker=False)
    sample_case2 = get_pce(age=55, is_male=False, is_african_american=True, has_ht_treatment=False, has_diabetes=False, total_chol=213, hdl_chol=50, systolic_bp=120, is_smoker=False)
    sample_case3 = get_pce(age=55, is_male=True, is_african_american=False, has_ht_treatment=False, has_diabetes=False, total_chol=213, hdl_chol=50, systolic_bp=120, is_smoker=False)
    sample_case4 = get_pce(age=55, is_male=True, is_african_american=True, has_ht_treatment=False, has_diabetes=False, total_chol=213, hdl_chol=50, systolic_bp=120, is_smoker=False)
    
if __name__=='__main__':
    main()

