library(tidyverse)
library(dplyr)
library(janitor)

dataPath <- '/PATH_TO/data/'
cohortPath1y <- '/PATH_TO/DATA/1y_traintestsplit.csv'
cohortPath5y <- '/PATH_TO/DATA/5y_traintestsplit.csv'

features <- read_csv(paste(dataPath,"all_ft_matrix.csv", sep="/"))
ft_matrix_1yr_imp <- read_csv(paste(dataPath,"1yr_ft_matrix_imputed.csv", sep="/"))
ft_matrix_5yr_imp <- read_csv(paste(dataPath,"5yr_ft_matrix_imputed.csv", sep="/"))

# calculation of CCI from ICD10 codes- based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6684052/
# 
mi_codes <- c('I21', 'I22', 'I25.2')
hf_codes <- c('I11.0', 'I13.0', 'I13.2', 'I25.5', 'I42.0', 'I42.5', 'I42.6', 'I42.7', 'I42.8', 'I42.9', 'I43', 'I50', 'P29.0')
pvd_codes <- c('I70', 'I71', 'I73.1', 'I73.8', 'I73.9', 'I77.1', 'I79.0','I79.1','I79.8','K55.1','K55.8','K55.9','Z95.8','Z95.9')
cvd_codes <- c('G45', 'G46', 'H34.0x','H34.2x', 'I60', 'I61', 'I62','I63', 'I64','I65','I66','I67','I68')
dementia_codes <- c('F01', 'F02', 'F03', 'F04', 'F05', 'F06.1', 'F06.8', 'G13.2', 'G13.8', 'G30', 'G31', 'G31.1', 'G31.2', 'G91.4', 'G94', 'R41.81', 'R54')
cpulmd_codes <- c('J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67', 'J68.4','J70.1', 'J70.3')
rheum_codes <- c('M05', 'M06', 'M31.5', 'M32', 'M33', 'M34', 'M35.1', 'M35.3', 'M36')
pep_ul_codes <- c('K25', 'K26', 'K27', 'K28')
liv_codes <- c('B18', 'K70.0', 'K70.1', 'K70.2', 'K70.3', 'K70.9', 'K71.3', 'K71.4', 'K71.5', 'K71.7', 'K73', 'K74', 'K76.0', 'K76.2', 'K76.3', 'K76.4', 'K76.8', 'K76.9', 'Z94.4')
diabetes_codes <- c('E08.0x', 'E08.1x', 'E08.6x', 'E08.8x', 'E08.9x',
                    'E09.0x', 'E09.1x', 'E09.6x', 'E09.8x', 'E09.9x',
                    'E10.0x', 'E10.1x', 'E10.6x', 'E10.8x', 'E10.9x',
                    'E11.0x', 'E11.1x', 'E11.6x', 'E11.8x', 'E11.9x'
                    'E13.0x', 'E13.1x', 'E13.6x', 'E13.8x', 'E13.9x')
kidney_codes <- c('I12.9', 'I13.0', 'I13.10', 'N03', 'N05', 'N18.1', 'N18.2', 'N18.3', 'N18.4', 'N18.9', 'Z94.0')
comp_diabetes_codes <- c('E08.2', 'E08.3', 'E08.4', 'E08.5',
                        'E09.2', 'E09.3', 'E09.4', 'E09.5',
                        'E10.2', 'E10.3', 'E10.4', 'E10.5',
                        'E11.2', 'E11.3', 'E11.4', 'E11.5',
                        'E13.2', 'E13.3', 'E13.4', 'E13.5')
plegia_codes <- c('G04.1', 'G11.4', 'G80.0', 'G80.1', 'G80.2', 'G81', 'G82', 'G83')
malign_codes <- c('C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31', 'C32', 'C33', 'C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','C60','C61','C62','C63','C76','C80.1','C81','C82','C83','C84','C85','C88','C97')
sev_liv_codes <- c('I85.0x', 'I86.4','K70.4x','K71.1x','K72.9x','K76.5','K76.6','K76.7')
sev_renal_codes <- c('I12.0','I13.11','I13.2','N18.5','N18.6','N19','N25.0','Z49','Z99.2')
hiv_codes <- c('B20')
mets_codes <- c('C77','C78','C79','C80','C80.2')
aids_codes <- c('B37','C53','B38','B45','A07.2','B25','G93.4x','B00','B39','A07.3','C46','C81','C82','C83','C84','C85','C88','C90','C91','C92','C93','C94','C95','C96','A31','A15','A16','A17','A18','A19','B59','Z87.01','A81.2','A02.1','B58','R64')

# Hemiplegia/paraplegia (Condition 13) trumps cerebrovascular disease (Condition 4)
# Liver disease, moderate-severe (Condition 5) trumps liver disease, mild (Condition 9)
# Diabetes with complications (Condition 12) trumps Diabetes, uncomplicated (Condition 10)
# Renal disease, severe (Condition 16) trumps Renal disease, mild-moderate (Condition 11)
# Metastatic solid tumor (Condition 18) trumps malignancy (Condition 14)
# AIDS (Condition 19) trumps HIV (Condition 14)
# Scores: 1-11: 1, 12-14: 2, 15-17:3, 18-19:6

