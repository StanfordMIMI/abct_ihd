library(tidyverse)
library(dplyr)
library(janitor)

dataPath <- '/PATH_TO/data/'
cohortPath1y <- '/PATH_TO/DATA/1y_traintestsplit.csv'
cohortPath5y <- '/PATH_TO/DATA/5y_traintestsplit.csv'

features <- read_csv(paste(dataPath,"all_ft_matrix.csv", sep="/"))
ft_matrix_1yr_imp <- read_csv(paste(dataPath,"1yr_ft_matrix_imputed.csv", sep="/"))
ft_matrix_5yr_imp <- read_csv(paste(dataPath,"5yr_ft_matrix_imputed.csv", sep="/"))

diabetes_dx <- tibble('cols' = names(ft_matrix_1yr_imp)) %>% 
  filter(str_detect(cols, "icd")) %>%
  mutate(code = str_remove(cols,"icd10_"),
         code_abb = str_sub(code, start=1, end=regexpr("\\.", code)-1)) %>%
  filter(code_abb %in% c("E10", "E11", "E12", "E13", "E14")) %>%
  select(cols) %>%
  pull

ht_meds <- read_csv(paste(dataPath, "ht_meds.csv", sep="/")) %>%
  clean_names

get_frs_features <- function(imputed_table, ht_meds){
  frs_features <- imputed_table %>%
    select(c(id, latest_value_chol_total, latest_value_chol_hdl, latest_value_systolic, smoker, 
             age_at_scan, gender, diabetes_dx)) %>%
    left_join(select(ht_meds, id, ht_meds), by="id") %>%
    mutate(diabetes = rowSums(select(., diabetes_dx)) > 0) %>%
    select(-diabetes_dx) %>%
    mutate(log_age = log(age_at_scan),
           log_total_chol = log(latest_value_chol_total),
           log_hdl = log(latest_value_chol_hdl),
           log_sbp = log(latest_value_systolic),
           smoker = if_else(smoker, 1, 0),
           treated_sbp = ht_meds,
           diabetes = if_else(diabetes, 1, 0)) %>%
    select(c(id, gender, treated_sbp, log_age,log_total_chol, log_hdl, log_sbp, smoker, diabetes))
  return (frs_features)
}
frs_features1yr <- get_frs_features(ft_matrix_1yr_imp, ht_meds)
frs_features5yr <- get_frs_features(ft_matrix_5yr_imp, ht_meds)


#From framingham study, table2: https://ahajournals.org/doi/full/10.1161/circulationaha.107.699579
# log_age, log_total_cholesterol, log_hdl_cholesterol, log_sbp_not_treated, log_sbp_treated, smoking,
#diabetes

wbu <- c(2.32888, 1.20904, -.70833, 2.76157, .52873, .69154) #women_betas_untreatedht
wbt <- c(2.32888, 1.20904, -.70833, 2.82263, .52873, .69154) #women_betas_treatedht
mbu <- c(3.06117, 1.12370, -.93263, 1.93303, .65451, .57367) #men_betas_untreatedht
mbt <- c(3.06117, 1.12370, -.93263, 1.99881, .65451, .57367) #men_betas_treatedht

means_w <- sum(c(2.32888, 1.20904, -.70833, 2.76157, 2.82263, .52873, .69154) * 
                 c(3.8686, 5.3504, 4.0176, 4.2400, .5826,.3423,.0376))
means_m <- sum(c(3.06117, 1.12370, -.93263, 1.93303, 1.99881, .65451, .57367) * 
                 c(3.8560, 5.3420, 3.7686, 4.3544, .5019, .3522, .0650))
get_frs <- function(frs_features){
  frs <- frs_features %>% 
    mutate(sum1 = ifelse(gender==1, 
                         if_else(treated_sbp,
                                 wbt[1]*log_age+wbt[2]*log_total_chol+ wbt[3]*log_hdl+wbt[4]*log_sbp+wbt[5]*smoker+wbt[6]*diabetes,
                                 wbu[1]*log_age+wbu[2]*log_total_chol+ wbu[3]*log_hdl+wbu[4]*log_sbp+wbu[5]*smoker+wbu[6]*diabetes),
                         if_else(treated_sbp,
                                 mbt[1]*log_age+mbt[2]*log_total_chol+ mbt[3]*log_hdl+mbt[4]*log_sbp+mbt[5]*smoker+mbt[6]*diabetes,
                                 mbu[1]*log_age+mbu[2]*log_total_chol+ mbu[3]*log_hdl+mbu[4]*log_sbp+mbu[5]*smoker+mbu[6]*diabetes)),
           sum2 = if_else(gender==1, means_w, means_m),
           frs = if_else(gender==1,
                         1-0.95012^exp(sum1-sum2),
                         1-0.88936^exp(sum1-sum2))) %>%
    select(c(id, log_age, frs))
  return (frs)
}

frs_1yr <- get_frs(frs_features1yr)
frs_5yr <- get_frs(frs_features5yr)

write_csv(frs_1yr, paste(dataPath, '1yr_FR_scores.csv', sep='/'))
write_csv(frs_5yr, paste(dataPath, '5yr_FR_scores.csv', sep='/'))