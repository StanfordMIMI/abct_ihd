library(tidyverse)
library(lubridate)
library(janitor)
library(tidyimpute)
library(readxl)

# Read data files
dataPath <- '/PATH_TO/data/anonymized'

cohortPath_1y <- paste(dataPath, 'IHD_8139_1y_ft_matrix_full.csv', sep='/')
cohortPath_5y <- paste(dataPath, 'IHD_8139_5y_ft_matrix_full.csv', sep='/')

data_1y <- read_csv(cohortPath_1y) 
data_5y <- read_csv(cohortPath_5y)

data_1y <- data_1y %>%
  mutate(set=if_else(set=='test','test','train'))

data_5y <- data_5y %>%
  mutate(set=if_else(set=='test','test','train'))


split_by_set_and_label <- function(data, ihd_labels) {
  train_control <-  filter(data, (set=="train")|(set=='val'), !label %in% ihd_labels)
  train_IHD <- filter(data, (set=="train")|(set=='val'), label %in% ihd_labels)
  test_control <- filter(data, set=="test", !label %in% ihd_labels)
  test_IHD <- filter(data, set=="test", label %in% ihd_labels)
  return (list(train_control, train_IHD, test_control, test_IHD))
}

split_data_1y <- split_by_set_and_label(data_1y, c(1))
split_data_5y <- split_by_set_and_label(data_5y, c(1,2))

get_var_summary_data <- function(table, var, percentage_or_mean){
  data_of_interest <- select(table, var) %>% pull
  missing = sum(is.na(data_of_interest))
  if(percentage_or_mean=="mean"){
    avg = mean(data_of_interest, na.rm=TRUE)
    sd = sd(data_of_interest, na.rm=TRUE)
    
    return (c(avg, sd, missing, length(data_of_interest)))
  }
  else{
    percent = 100*sum(data_of_interest == percentage_or_mean, na.rm=TRUE)/(length(data_of_interest)-missing)
    return (c(percent, missing, length(data_of_interest)))
  }
}
get_summary_data <- function(data, labels, var, percentage_or_mean){
  split_data <- split_by_set_and_label(data, labels)
  train_control <- split_data[[1]]
  train_IHD <- split_data[[2]]
  test_control <- split_data[[3]]
  test_IHD <- split_data[[4]]
  
  print(get_var_summary_data(train_control, var, percentage_or_mean))
  print(get_var_summary_data(train_IHD, var, percentage_or_mean))
  print(get_var_summary_data(test_control, var, percentage_or_mean))
  print(get_var_summary_data(test_IHD, var, percentage_or_mean))
}
get_summary_data(data_1y, c(1), "age_at_scan", "mean")
get_summary_data(data_1y, c(1), "gender", TRUE)
get_summary_data(data_1y, c(1), "smoker", TRUE)
get_summary_data(data_1y, c(1), "latest_value_chol_hdl", "mean")
get_summary_data(data_1y, c(1), "latest_value_chol_total", "mean")
get_summary_data(data_1y, c(1), "latest_value_systolic", "mean")

get_summary_data(data_5y, c(1,2), "age_at_scan", "mean")
get_summary_data(data_5y, c(1,2), "gender", TRUE)
get_summary_data(data_5y, c(1,2), "smoker", TRUE)
get_summary_data(data_5y, c(1,2), "latest_value_chol_hdl", "mean")
get_summary_data(data_5y, c(1,2), "latest_value_chol_total", "mean")
get_summary_data(data_5y, c(1,2), "latest_value_systolic", "mean")


diabetes_dx <- tibble('cols' = names(data_1y)) %>% 
  filter(str_detect(cols, "icd")) %>%
  mutate(code = str_remove(cols,"icd10_"),
         code_abb = str_sub(code, start=1, end=regexpr("\\.", code)-1)) %>%
  filter(code_abb %in% c("E10", "E11", "E12", "E13", "E14")) %>%
  select(cols) %>%
  pull

#Diabetes
data_1y %>% 
  select(anon_id, label, set, diabetes_dx) %>% mutate(label = if_else(label%in%c(1),1,0)) %>%
  mutate(diabetes = rowSums(select(., diabetes_dx)) > 0) %>%
  select(-diabetes_dx) %>%
  mutate(set=if_else(set=='test','test','train')) %>%
  group_by(label,set, diabetes) %>%
  summarise(n=n()) %>% distinct %>% mutate(freq = n / sum(n)) %>% 
  arrange(desc(set)) %>%
  filter(diabetes)

data_5y %>% 
  select(anon_id, label, set, diabetes_dx) %>% mutate(label = if_else(label%in%c(1,2),1,0)) %>%
  mutate(diabetes = rowSums(select(., diabetes_dx)) > 0) %>%
  select(-diabetes_dx) %>%
  group_by(label,set, diabetes) %>%
  summarise(n=n()) %>% distinct %>% mutate(freq = n / sum(n)) %>% 
  arrange(desc(set)) %>%
  filter(diabetes)

#import additional for PCE specific features
ht_meds <- read_csv(paste(dataPath, "IHD_8139_ht_meds.csv", sep="/"))

data_1y %>%
  select(anon_id, label, set) %>%
  mutate(label = if_else(label%in%c(1),1,0)) %>%
  left_join(ht_meds, by=c("anon_id"="anon_id")) %>%
  mutate(set=if_else(set=='test','test','train')) %>%
  group_by(label,set, ht_meds) %>% summarise(n=n()) %>% distinct %>% mutate(freq = n / sum(n)) %>% 
  arrange(desc(set)) %>%
  filter(ht_meds)

data_5y %>%
  select(anon_id, label, set) %>%
  mutate(label = if_else(label%in%c(1,2),1,0)) %>%
  left_join(ht_meds, by=c("anon_id"="anon_id")) %>%
  mutate(set=if_else(set=='test','test','train')) %>%
  group_by(label,set, ht_meds) %>% summarise(n=n()) %>% distinct %>% mutate(freq = n / sum(n)) %>% 
  arrange(desc(set)) %>%
  filter(ht_meds)


# Features calculated from L3 slice
seg_fts <- read_csv("/PATH_TO/IHD_8139_segmentation_fts.csv") %>%
  select(anon_id, muscle_HU, vat_sat_ratio)

get_summary_data(data_1y %>% select(anon_id, set, label) %>% left_join(seg_fts, by="anon_id"), c(1), "vat_sat_ratio", "mean")
get_summary_data(data_1y %>% select(anon_id, set, label) %>% left_join(seg_fts, by="anon_id"), c(1), "muscle_HU", "mean")

get_summary_data(data_5y %>% select(anon_id, set, label) %>% left_join(seg_fts, by="anon_id"), c(1,2), "vat_sat_ratio", "mean")
get_summary_data(data_5y %>% select(anon_id, set, label) %>% left_join(seg_fts, by="anon_id"), c(1,2), "muscle_HU", "mean")



# Get demographics including race

demographics <- get_all_demographics() %>%
  mutate(race_eth = if_else(ethnicity=="Hispanic/Latino", "Hispanic", race),
         race_eth = if_else(race_eth %in% c("Asian","Black", "Hispanic","White"), race_eth, "Other"))

data_1y %>%
  select(anon_id, set, label) %>%
  mutate(label = if_else(label%in%c(1),1,0)) %>%
  left_join(demographics, by=c("anon_id")) %>%
  group_by(label,set, race_eth) %>% summarise(n=n()) %>% distinct %>% mutate(freq = n / sum(n)) %>%
  arrange(desc(set), race_eth, label)

data_5y %>%
  select(anon_id, set, label) %>%
  mutate(label = if_else(label%in%c(1),1,0)) %>%
  left_join(demographics, by=c("anon_id")) %>%
  group_by(label,set, race_eth) %>% summarise(n=n()) %>% distinct %>% mutate(freq = n / sum(n)) %>%
  arrange(desc(set), race_eth, label)

