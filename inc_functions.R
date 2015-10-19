######################################################### 
# 
# THIS SCRIPT CONTAINS THE FUNCTIONS REQUIRED BY MAIN.R
# 
# 

#### PREDEDINED COLUMN TYPES
factor.cols   <- c('STUDYID',	'GLEAS_DX',	'TSTAG_DX',	'RACE_C',	'REGION_C',	'SMOKE',	'SMOKFREQ',	
                    'SMOKSTAT',	'ECOG_C',	'TRT2_ID',	'TRT3_ID',	'NON_TARGET',	'TARGET',	'BONE',	'RECTAL',	'LYMPH_NODES',	
                    'KIDNEYS',	'LUNGS',	'LIVER',	'PLEURA',	'OTHER',	'PROSTATE',	'ADRENAL',	'BLADDER',	'PERITONEUM',	
                    'COLON',	'SOFT_TISSUE',	'ABDOMINAL',	'ORCHIDECTOMY',	'PROSTATECTOMY',	'TURP',	'LYMPHADENECTOMY',	
                    'SPINAL_CORD_SURGERY',	'BILATERAL_ORCHIDECTOMY',	'PRIOR_RADIOTHERAPY',	'ANALGESICS',	'ANTI_ANDROGENS',	
                    'GLUCOCORTICOID',	'GONADOTROPIN',	'BISPHOSPHONATE',	'CORTICOSTEROID',	'IMIDAZOLE',	'ACE_INHIBITORS',	
                    'BETA_BLOCKING',	'HMG_COA_REDUCT',	'ESTROGENS',	'ANTI_ESTROGENS',	'ARTTHROM',	'CEREBACC',	'CHF',	
                    'DVT',	'DIAB',	'GASTREFL',	'GIBLEED',	'MI',	'PUD',	'PULMEMB',	'PATHFRAC',	'SPINCOMP',	'COPD',	
                    'MHBLOOD',	'MHCARD',	'MHCONGEN',	'MHEAR',	'MHENDO',	'MHEYE',	'MHGASTRO',	'MHGEN',	'MHHEPATO',	
                    'MHIMMUNE',	'MHINFECT',	'MHINJURY',	'MHINVEST',	'MHMETAB',	'MHMUSCLE',	'MHNEOPLA',	'MHNERV',	'MHPSYCH',	
                    'MHRENAL',	'MHRESP',	'MHSKIN',	'MHSOCIAL',	'MHSURG',	'MHVASC', 'CancerSums', 'MedBodySums')

binary.cols   <- c('NON_TARGET',	'TARGET',	'BONE',	'RECTAL',	'LYMPH_NODES',	'KIDNEYS',	'LUNGS',	'LIVER',	'PLEURA',	
                 'OTHER',	'PROSTATE',	'ADRENAL',	'BLADDER',	'PERITONEUM',	'COLON',	'HEAD_AND_NECK',	'SOFT_TISSUE',	
                 'STOMACH',	'PANCREAS',	'THYROID',	'ABDOMINAL',	'ORCHIDECTOMY',	'PROSTATECTOMY',	'TURP',	'LYMPHADENECTOMY',	
                 'SPINAL_CORD_SURGERY',	'BILATERAL_ORCHIDECTOMY',	'PRIOR_RADIOTHERAPY',	'ANALGESICS',	'ANTI_ANDROGENS',	
                 'GLUCOCORTICOID',	'GONADOTROPIN',	'BISPHOSPHONATE',	'CORTICOSTEROID',	'IMIDAZOLE',	'ACE_INHIBITORS',	
                 'BETA_BLOCKING',	'HMG_COA_REDUCT',	'ESTROGENS',	'ANTI_ESTROGENS',	'ARTTHROM',	'CEREBACC',	'CHF',	'DVT',	
                 'DIAB',	'GASTREFL',	'GIBLEED',	'MI',	'PUD',	'PULMEMB',	'PATHFRAC',	'SPINCOMP',	'COPD',	'MHBLOOD',	'MHCARD',	
                 'MHCONGEN',	'MHEAR',	'MHENDO',	'MHEYE',	'MHGASTRO',	'MHGEN',	'MHHEPATO',	'MHIMMUNE',	'MHINFECT',	'MHINJURY',	
                 'MHINVEST',	'MHMETAB',	'MHMUSCLE',	'MHNEOPLA',	'MHNERV',	'MHPSYCH',	'MHRENAL',	'MHRESP',	'MHSKIN',	'MHSOCIAL',	
                 'MHSURG',	'MHVASC','ClustLab', 'ClustMedHistory')
numeric.cols  <- c('ALP',	'ALT',	'AST',	'CA',	'CREAT',	'HB',	'LDH',	'NEU',	'PLT',	'PSA',	'TBILI',	'TESTO',	'WBC',	
                  'CREACL',	'NA',	'MG',	'PHOS',	'ALB',	'TPRO',	'RBC',	'LYM',	'BUN',	'CCRC',	'GLU',	'CREACLCA')


## FUNCTION AS USED FOR GREEDY METHOD IN FEATURE SELECTIONS

#' evaluatorrf(subset.feature) 
#' As used by Hill-Climbing algorithm to evaluate Feature Selection. 
#' @PARAM ARGS: taking a set of features (vector) to evaluate
#' @RETURN: AUC 
#' 


evaluatorrf <- function(subset.feature) {
  k <- 3              # Number of Crossfold Validation for this particular evaluation (bigger=more stable, smaller=faster)
  results = sapply(1:k, function(i) {
    bag0.subset     <- bag0[sample(nrow(bag0), 
                                   round(nrow(bag0)*baggingRatio), replace = F),]
    tmp             <- as.data.frame(rbind(bag1,bag0.subset))
    model.rf        <- randomForest(DISCONT ~., data=tmp[,c("DISCONT",subset.feature)], mtry=round(length(subset.feature)/3), na.action = na.omit, probability=T)
    result.rf       <- predict(model.rf,   curr.testing.data, type = "prob")[,2]
    rf.auc <- score_q2(result.rf , curr.testing.data$DISCONT)
    return(rf.auc)
  })
  print(paste("RF, FS - Climbing with current AUC:",mean(results)))
  return(mean(results))
}



#' get.training.data(fold)
#' Split the Cleaned-Up training data based on the study
#' @PARAM ARGS: numeric: 1,2, or 3
#' @RETURN
#' Fold =1 Output a dataframe containing EFC6546, and CELGENE
#' Fold =2 Output a dataframe containing ACCENT2, and CELGENE
#' Fold =3 Output a dataframe containing EFC6546, and ACCENT2
#'

get.training.data <- function(fold)
{
  if (fold == 1)
  {
    training.cols = c(4,3)
    testing.col   = c(1) 
    testing.name  = "ACCENT2"
    
  } else if (fold == 2) {
    training.cols  = c(1,3)
    testing.col    = c(4) 
    testing.name   = "EFC6546"
    
  } else {
    training.cols = c(1,4)
    testing.col   = c(3) 
    testing.name  = "CELGENE"
  }
  
  ## Split between training and testing
  curr.training.data <- as.data.frame(rbind(split.table[[training.cols[1]]], split.table[[training.cols[2]]]))
  curr.testing.data  <- as.data.frame(split.table[[testing.col]])
  return(curr.training.data)
}


#' get.testing.data(fold)
#' Split the Cleaned-Up training data based on the study
#' @PARAM ARGS: numeric: 1,2, or 3
#' @RETURN
#' Fold =1 Output a dataframe containing ACCENT2
#' Fold =2 Output a dataframe containing EFC6546
#' Fold =3 Output a dataframe containing EFC6546
#' 

get.testing.data <- function(fold)
{
  if (fold == 1)
  {
    training.cols = c(4,3)
    testing.col   = c(1) 
    testing.name  = "ACCENT2"
    
  } else if (fold == 2) {
    training.cols  = c(1,3)
    testing.col    = c(4) 
    testing.name   = "EFC6546"
    
  } else {
    training.cols = c(1,4)
    testing.col   = c(3) 
    testing.name  = "CELGENE"
  }
  
  ## Split between training and testing
  curr.training.data <- as.data.frame(rbind(split.table[[training.cols[1]]], split.table[[training.cols[2]]]))
  curr.testing.data  <- as.data.frame(split.table[[testing.col]])
  return(curr.testing.data)
}


#' 
#' get.testing.name(fold)
#' Returning the target (testing data) name of given fold
#' @PARAM ARGS: numeric: 1,2, or 3
#' @RETURN
#' Fold =1  "ACCENT2"
#' Fold =2  "EFC6546"
#' Fold =3  "EFC6546"

get.testing.name <- function(fold)
{
  if (fold == 1)
  {
    testing.name  = "ACCENT2"
  } else if (fold == 2) {
    testing.name   = "EFC6546"
  } else {
    testing.name  = "CELGENE"
  }
  return(testing.name)
}