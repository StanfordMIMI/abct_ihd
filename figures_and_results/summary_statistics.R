library(tidyverse)


mergePath <- '/PATH_TO/data'
dataPath <- '/PATH_to/data'

cohortPath_1y <- paste(mergePath, 'IHD_8139_1y_ft_matrix_full.csv', sep='/')
cohortPath_5y <- paste(mergePath, 'IHD_8139_5y_ft_matrix_full.csv', sep='/')

data_1y <- read_csv(cohortPath_1y) 
data_5y <- read_csv(cohortPath_5y)

data_1y <- data_1y %>%
  mutate(set=if_else(set=='test','test','train'))

data_5y <- data_5y %>%
  mutate(set=if_else(set=='test','test','train'))


#num images
data_5y %>% summarize(n_pats = n_distinct(accession))
#num patients
data_5y %>% summarize(n_pats = n_distinct(mrn))
#follow up - last encounter from cohort_check.R
last_encounter %>% right_join(select(data_1y, accession)) %>% summarize(avg_fu = mean(as.numeric(ct_to_last_enc_date))/365.25, iqr_fu = IQR(ct_to_last_enc_date)/365.25)

# % pos
table(data_1y$label)
data_1y %>% summarize(pos = mean(label))
table(data_5y$label)
data_5y %>% summarize(pos = mean(label))

#avg age
data_1y %>% select(age_at_scan) %>% summarize(avg = mean(age_at_scan), sd = sd(age_at_scan))
# % men
data_1y %>% summarize(men = mean(gender))

#correctness of L3 selection model
l3_slice_path <- "/PATH_TO/predictions/slice_predictions/checked_final.csv"

l3_slice_preds <- read_csv(l3_slice_path) %>% 
  select(accession, is_incorrect, pred_y, corrected_pred_y) %>%
  right_join(select(data_1y, accession))
table(l3_slice_preds$is_incorrect)

