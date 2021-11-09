
library(tidyverse)
library(readxl)
library(janitor)

cohortPath1y <- '/PATH_TO/data/1y_traintestsplit.csv'
cohortPath5y <- '/PATH_TO/data/5y_traintestsplit.csv'
dataPath <- '/PATH_TO/data/'
featurePath1y <- paste(dataPath,"1yr_ft_matrix_imputed.csv", sep="/")
featurePath5y <- paste(dataPath,"5yr_ft_matrix_imputed.csv", sep="/")


# Step 1. determine ethnicity

 get_ethnicity_data <- function(cohortPath){
   ethnicity_data <- read_csv(paste(dataPath, 'ethnicity_data.csv')) %>%
    return(ethnicity_data)
 }

#Step 2. Calculate other PCE features
 
 get_pce_features <- function(featurePath){
   features <- read_csv(featurePath)
   
   diabetes_dx <- tibble('cols' = names(features)) %>% 
     filter(str_detect(cols, "icd")) %>%
     mutate(code = str_remove(cols,"icd10_"),
            code_abb = str_sub(code, start=1, end=regexpr("\\.", code)-1)) %>%
     filter(code_abb %in% c("E10", "E11", "E12", "E13", "E14")) %>%
     select(cols) %>%
     pull
   
   ht_meds <- read_csv(paste(dataPath, "ht_meds.csv", sep="/")) %>%
     clean_names
   
   pce_features <- features %>%
     select(c(id, latest_value_chol_total, latest_value_chol_hdl, latest_value_systolic, smoker, 
              age_at_scan, gender, diabetes_dx)) %>%
     mutate(diabetes = rowSums(select(., diabetes_dx)) > 0) %>%
     select(-diabetes_dx) %>%
     inner_join(ht_meds, by=c("id")) %>%
     mutate(log_age = log(age_at_scan),
            log_total_chol = log(latest_value_chol_total),
            log_hdl = log(latest_value_chol_hdl),
            log_sbp = log(latest_value_systolic),
            smoker = if_else(smoker, 1, 0),
            treated_sbp = ht_meds,
            diabetes = if_else(diabetes, 1, 0)) %>%
     select(c(id, gender, treated_sbp, log_age,log_total_chol, log_hdl, log_sbp, smoker, diabetes))
   return(pce_features)
 }

 #Step 3. Define PCE functions

get_pce_risk <- function(pce_features, demographics){
  
  #log_age, log_age^2, log_total_chol, log_age_x_total_chol, log_hdl, log_age_x_hdl,
  #log_sbp_treated, log_age_x_log_sbp_treated, log_untreated_sbp, log_age_x_log_sbp_untreated,
  #smoker, log_age_x_smoker, diabetes
  pce_risk_coefs_ww <- c(-29.799, 4.884, 13.540, -3.114, -13.578, 3.149, 2.019, 0, 1.957,
                         0, 7.574, -1.665, 0.661)
  ww_mean_baseline <- c(-29.1817, 0.96652)
  pce_risk_coefs_bw <- c(17.1141, 0, 0.9396, 0, -18.9196, 4.4748, 29.2907, -6.4321, 27.8197, -6.0873, 0.6908, 0, 
                         0.8738)
  bw_mean_baseline <- c(86.6081, 0.95334)
  
  #log_age, log_total_chol, log_age_x_total_chol, log_hdl, log_age_x_hdl,
  #log_sbp_treated, log_untreated_sbp, smoker, log_age_x_smoker, diabetes
  pce_risk_coefs_wm <- c(12.344, 11.853, -2.664, -7.990, 1.769, 1.797, 1.764, 7.837, -1.795, 0.658)
  wm_mean_baseline <- c(61.1816, 0.91436)
  pce_risk_coefs_bm <- c(2.469, 0.302, 0, -0.307, 0, 1.916, 1.809, 0.549, 0, 0.645)
  bm_mean_baseline <- c(19.5425, 0.89536)
  
  pce_risk_w <- pce_features %>% 
    left_join(select(demographics, id, race4calc), by=c("id")) %>%
    filter(!gender) %>%
    mutate(log_age2 = log_age^2,
           log_age_x_total_chol = log_age * log_total_chol,
           log_age_x_hdl = log_age * log_hdl,
           log_sbp_treated = if_else(treated_sbp, log_sbp, 0),
           log_age_x_log_sbp_treated = if_else(treated_sbp, log_age*log_sbp, 0),
           log_untreated_sbp = if_else(!treated_sbp, log_sbp, 0),
           log_age_x_log_sbp_untreated = if_else(!treated_sbp, log_age*log_sbp, 0),
           log_age_x_smoker = log_age * smoker,
           #risk score calculations:
           sum = if_else(race4calc == "Black",
                         pce_risk_coefs_bw[1]*log_age+pce_risk_coefs_bw[2]*log_age2+pce_risk_coefs_bw[3]*log_total_chol+
                           pce_risk_coefs_bw[4]*log_age_x_total_chol+pce_risk_coefs_bw[5]*log_hdl+
                           pce_risk_coefs_bw[6]*log_age_x_hdl+pce_risk_coefs_bw[7]*log_sbp_treated+
                           pce_risk_coefs_bw[8]*log_age_x_log_sbp_treated+pce_risk_coefs_bw[9]*log_untreated_sbp+
                           pce_risk_coefs_bw[10]*log_age_x_log_sbp_untreated +pce_risk_coefs_bw[11]*smoker+
                           pce_risk_coefs_bw[12]*log_age_x_smoker+ pce_risk_coefs_bw[13]*diabetes,
                         pce_risk_coefs_ww[1]*log_age+pce_risk_coefs_ww[2]*log_age2+pce_risk_coefs_ww[3]*log_total_chol+
                           pce_risk_coefs_ww[4]*log_age_x_total_chol+pce_risk_coefs_ww[5]*log_hdl+
                           pce_risk_coefs_ww[6]*log_age_x_hdl+pce_risk_coefs_ww[7]*log_sbp_treated+
                           pce_risk_coefs_ww[8]*log_age_x_log_sbp_treated+pce_risk_coefs_ww[9]*log_untreated_sbp+
                           pce_risk_coefs_ww[10]*log_age_x_log_sbp_untreated +pce_risk_coefs_ww[11]*smoker+
                           pce_risk_coefs_ww[12]*log_age_x_smoker+ pce_risk_coefs_ww[13]*diabetes),
           pce_risk = if_else(race4calc == "Black",
                              1-bw_mean_baseline[2]^exp(sum-bw_mean_baseline[1]),
                              1-ww_mean_baseline[2]^exp(sum-ww_mean_baseline[1])))   
  
  
  
  pce_risk_m <- pce_features %>% 
    left_join(select(demographics, id, race4calc), by=c("id")) %>%
    filter(gender) %>%
    mutate(log_age_x_total_chol = log_age * log_total_chol,
           log_age_x_hdl = log_age * log_hdl,
           log_sbp_treated = if_else(treated_sbp, log_sbp, 0),
           log_untreated_sbp = if_else(!treated_sbp, log_sbp, 0),
           log_age_x_smoker = log_age * smoker,
           #risk score calculations:
           sum = if_else(race4calc == "Black",
                         pce_risk_coefs_bm[1]*log_age+pce_risk_coefs_bm[2]*log_total_chol+
                           pce_risk_coefs_bm[3]*log_age_x_total_chol+pce_risk_coefs_bm[4]*log_hdl+
                           pce_risk_coefs_bm[5]*log_age_x_hdl+ pce_risk_coefs_bm[6]*log_sbp_treated+
                           pce_risk_coefs_bm[7]*log_untreated_sbp+ pce_risk_coefs_bm[8]*smoker+ 
                           pce_risk_coefs_bm[9]*log_age_x_smoker+ pce_risk_coefs_bm[10]*diabetes,
                         pce_risk_coefs_wm[1]*log_age+pce_risk_coefs_wm[2]*log_total_chol+
                           pce_risk_coefs_wm[3]*log_age_x_total_chol+pce_risk_coefs_wm[4]*log_hdl+
                           pce_risk_coefs_wm[5]*log_age_x_hdl+ pce_risk_coefs_wm[6]*log_sbp_treated+
                           pce_risk_coefs_wm[7]*log_untreated_sbp+ pce_risk_coefs_wm[8]*smoker+ 
                           pce_risk_coefs_wm[9]*log_age_x_smoker+ pce_risk_coefs_wm[10]*diabetes
           ),
           pce_risk = if_else(race4calc == "Black",
                              1-bm_mean_baseline[2]^exp(sum-bm_mean_baseline[1]),
                              1-wm_mean_baseline[2]^exp(sum-wm_mean_baseline[1])))   
  
  pce_risk <- pce_risk_w %>%
    bind_rows(pce_risk_m) 
  return (pce_risk)
}

#Step 4. Get pce scores for desired cohorts and save 

#Cohort 1yr
dem_1yr <- get_ethnicity_data(cohortPath1y)
pce_ft_1yr <- get_pce_features(featurePath1y)

pce_risk_1yr <- get_pce_risk(pce_ft_1yr, dem_1yr)

pce_risks_1yr <- pce_risk_1yr %>%
  select(id, pce_risk)
write_csv(pce_risks_1yr, paste(dataPath, '1yr_PCE_scores.csv', sep='/'))

#Cohort 5yr
dem_5yr <- get_ethnicity_data(cohortPath5y)
pce_ft_5yr <- get_pce_features(featurePath5y)

pce_risk_5yr <- get_pce_risk(pce_ft_5yr, dem_5yr)

pce_risks_5yr <- pce_risk_5yr %>%
  select(id, pce_risk)
write_csv(pce_risks_5yr, paste(dataPath, '5yr_PCE_scores.csv', sep='/'))
