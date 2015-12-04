################################################################################################################ 
#                     DREAM CHALLENGES SUBMISSION (CHALLENGE 2.B)
# 
#                       TEAM YODA | UNIVERSITY OF WASHINGTON TACOMA
#                                 kayee@u.washington.edu
#                                     SUMMER - 2015
#
#             Predicting Discontinuation of Docetaxel Treatment for Metastatic 
#   Castration-Resistant Prostate Cancer (mCRPC) with hill-climbing and Random Forest
# 
#   Authors Statement
#   Kevin Anderson, Seyed Khankhajeh, Daniel Kristiyanto, Kaiyuan Shi, Seth West contributed equally as first authors. 
# 
#   Kevin Anderson, Seyed Khankhajeh, Daniel Kristiyanto, Kaiyuan Shi, Seth West, Azu Lee, Qi Wei, Migao Wu, Yunhong Yin 
#   conducted empirical studies comparing the performance of various machine learning and feature selection algorithms 
#   on this dataset.  
# 
#   Daniel Kristiyanto served as the team captain and managed the submission uploads.  
#   Ka Yee Yeung served as the PI, used this DREAM 9.5 challenge as a class project, provided guidance and co-ordinated  
#   this submission effort.
# 
#
###########################################################################################

# ADJUST-ACCORDINGLY (IFNECESSARY)
path.script = "~/Workspaces/git_repos/DREAM9.5/"
setwd(path.script)

require(FSelector) 
require(e1071)
require(timeROC)
require(caret)
require(unbalanced)
require(impute)
require(randomForest)
library(FSelector) 
library(e1071)
library(impute)
library(caret)
library(randomForest)
library(mlr)
library(unbalanced)


source(paste0(path.script,"inc_datacleanup.r"))     # Contains functions to clean up the data
source(paste0(path.script,"score.R"))               # Contains Score.R from Synapse as scoring
source(paste0(path.script,"inc_functions.R"))       # Contains various functions (evaluator, etc)

###### GLOBAL VARIALBLES ##############################################################
## GLOBAL VARIABLES AND PARAMETERS

CV                    = c(rep(1:5))    # Number of Crossfold Validation, each fold takes about 40 mins 
CV <- 1
# CV                    = seq(3,110,10)
# Itterate through every study with following target 
# (and use the other two as training data) 
# 1. ACCENT2, 2. EFC6546, 3. CELGENE 4. ALL COMBINED
STUDY                 = c(1,2,3)  
# STUDY                 = c(4)        # Enable to perform prediction for the Core_Validation Dataset  

## MODEL TUNING
FOLD.RATIO            = 0.9       # How many goes as training data, for STUDY= 4 ONLY
baggingRatio          = 0.9       # Used by Feature Selection Evaluation.
balace.ratio          = 31

rf.ntree              = 180
rf.mtry               = 4
k                     = 6
# k = 43

x.axis                = "Cross-fold Validation"

####### VARIABLE LIST ##############################################################
# IN GENERAL, HERE'S THE VARIABLES
core_table_training   <- NULL            # ORIGINAL CORE TRAINING  (RAW)
core_table_validation <- NULL            # ORIGINAL CORE TESTING  (RAW)
table.for.model       <- NULL            # CLEANED, AUGMENTED CORE TRAINING
table.for.validation  <- NULL            # CLEANED, AUGMENTED CORE TESTING
curr.training.data    <- NULL            # A SUBSET OF CURRENT FOLD'S TRAINING DATA
curr.testing.data     <- NULL            # A SUBSET OF CURRENT FOLD'S TESTING DATA
SCORING.TABLE         <- NULL            # SCORE RESULTS 
OUTPUT.TABLE          <- NULL            # SUBSET OF CURR.TESTING DATA WITH PROB RESULT
FINAL.TABLE           <- NULL            # CORE TESTING/VALIDATION FROM SYNAPSE WITH PREDICTION

# 
####### PREPROCESS ##############################################################
## GET THE DATA
core_table_training     <- read.csv("./DATA/CoreTable_training.csv"  , stringsAsFactors = F)
core_table_validation1  <- read.csv("./DATA/CoreTable_validation.csv", stringsAsFactors = F)
core_table_validation2  <- read.csv("./DATA/CoreTable_leaderboard.csv", stringsAsFactors = F)
core_table_validation   <- rbind(core_table_validation1, core_table_validation2)

## GET THE DATA CLEANED UP
#core_table_validation$DISCONT <- 0
combined_data                 <- rbind(core_table_training, core_table_validation)
clean_data                    <- dream9.cleanData(combined_data)

## SOME DATA CLEAN UP
clean_data$DEATH[clean_data$DEATH == "."]           <- NA
clean_data$DEATH[clean_data$DEATH == "YES"]         <- 1
clean_data$DEATH[clean_data$DEATH == ""]            <- 0

clean_data$LKADT_P[clean_data$LKADT_P == "."]       <- NA
clean_data$ENDTRS_C[clean_data$ENDTRS_C == "."]     <- "UNKNOWN"
clean_data$ENTRT_PC[clean_data$ENTRT_PC == "."]     <- NA
clean_data$PER_REF[clean_data$PER_REF == "."]       <- NA
clean_data$LKADT_REF[clean_data$LKADT_REF == "."]   <- NA
clean_data$ENTRT_PC[clean_data$ENTRT_PC == "."]     <- NA

####### EXTRAPOLATE ##############################################################
dependant.cols                <- c("DEATH", "LKADT_P", "ENDTRS_C", "ENTRT_PC", "PER_REF", "LKADT_REF","TSTAG_DX")
clean_data                    <- clean_data[,!(colnames(clean_data) %in% dependant.cols)]

## VARIABLE EXTRAPOLATE / AUGMENT
# to.explode                    <- c("AGEGRP2","RACE_C","REGION_C","SMOKE","SMOKFREQ","SMOKSTAT","ECOG_C","TRT2_ID","TRT3_ID")
to.explode                    <- c("TRT2_ID","SMOKE","SMOKFREQ","SMOKSTAT")
tmp                           <- clean_data[,to.explode]
f                             <- paste("~0+",paste(to.explode, collapse = "+"))
tmp                           <- model.matrix(as.formula(f), data=tmp)
tmp                           <- apply(tmp,2,factor)
clean_data                    <- cbind(clean_data,tmp)
factor.cols                   <- union(colnames(tmp),factor.cols)
binary.cols                   <- union(colnames(tmp),binary.cols)
to.drop                       <- c(to.explode,"TRT3_ID","GLEAS_DX","NON_TARGET", "ABDOMINAL","BONE","WEIGHTBL","HEIGHTBL")
clean_data                    <- clean_data[,!(colnames(clean_data) %in% to.drop)]
row.names(clean_data)         <- clean_data$RPT
halabi <- c("RACE_C","AGEGRP2","BMI","PRIOR_RADIOTHERAPY","ANALGESICS","ECOG_C","GLEAS_DX","ALB","LDH", "WBC",
            "AST","TBILI","PLT", "HB","ALT","TESTO","PSA","ALP")
halabi <- c("RACE_C","AGEGRP2","BMI","PRIOR_RADIOTHERAPY","ANALGESICS","ECOG_C","ALB","LDH", "WBC",
            "AST","TBILI","PLT", "HB","ALT","PSA","ALP")

halabi <- c(halabi, 
            "RECTAL",
            "LYMPH_NODES",
            "KIDNEYS",
            "LUNGS",
            "LIVER",
            "PLEURA",
            "OTHER",
            "PROSTATE",
            "ADRENAL",
            "BLADDER",
            "PERITONEUM",
            "COLON",
            "SOFT_TISSUE"
           )

# "HEAD_AND_NECK",
# "STOMACH",
# "PANCREAS",
# "THYROID"

## SPLIT THE DATA BACK TO TRAINING AND VALIDATION
removed.from.validation       <- c("DISCONT","WEIGHTBL","HEIGHTBL")
table.for.validation          <- clean_data[clean_data$STUDYID == "AZ",!(colnames(clean_data) %in% removed.from.validation)]
table.for.model               <- clean_data[((clean_data$STUDYID != "AZ") & !is.na(clean_data$DISCONT)), ]


## DROP DEPENDANT VARIABLES AWAY
to.drop                     <- c("LKADT_P","DEATH","PER_REF","LKADT_REF","LKADT_PER", "TSTAG_DX")
table.for.model             <- table.for.model[,!(colnames(table.for.model) %in% to.drop)]

# table.for.validation        <- clean_validation
split.table                 <- split(table.for.model, table.for.model$STUDYID)

####### CROSSFOLD START ##############################################################

for(cv in CV) #Begin Cross-Fold for Validation or for Model Tuning
{
# Unless tuning is performed, these vaiables should be disabled
# rf.ntree              = cv
# balace.ratio          = cv
# rf.mtry               = cv
# k                     = cv
library(caret)
train.index <- createDataPartition(table.for.model$DISCONT, p = FOLD.RATIO,list = FALSE, times = 1)
detach(package:caret, unload=T, force=T) 
SCORING.TABLE <- NULL
for (curr.study in STUDY)  ## LOOP THROUGH THE DATA FRAMES
{
  ## RETRIEVE THE CLEAN DATA SET 
  ## THE FUNCTION ASSUMES THAT THE CORE DATA IS ALREADY LOADED
  curr.training.data <- as.data.frame(get.training.data(fold=curr.study))
  curr.testing.data  <- as.data.frame(get.testing.data(fold=curr.study))
  testing.name       <- get.testing.name(curr.study)
  
  
  testing.name       <- get.testing.name(curr.study)
  

    
  if (curr.study == 4 ){         #  USE ALL FOR TESTING/TRAINING
    # set.seed(123) # Tuning? Normal: Disabled #####
    curr.training.data <- table.for.model[train.index,]
    curr.testing.data  <- table.for.model[-train.index,]
    testing.name      <- paste("ALL-fold")
  }
  curr.training.data <- as.data.frame(rbind(curr.training.data,curr.testing.data))
  
  
  ## DROP VARIABLES THAT ARE NOT AVAILABLE IN TESTING FROM TRAINING AND SOME REDUDANT
#   to.drop <- union(to.drop, c("SMOKFREQ",	"SMOKSTAT", "HEIGHTBL", "WEIGHTBL",	"WEIGHT", "NON_TARGET", "AGE", "X"))
#   tmp     <- curr.testing.data[curr.testing.data$DISCONT==1,]
#   tmp     <- tmp[,-c(1,2,3)]
#   tmp     <- sapply(tmp, var)
#   tmp     <- tmp[tmp==0]
#   to.drop <- union(to.drop, attr(tmp,"names"))
#   curr.training.data <- curr.training.data[,!names(curr.training.data) %in% to.drop]
#   
  
  ## UPDATE THE LIST OF VARIABLE GROUPS
  all.features          <- intersect(colnames(curr.training.data),colnames(curr.testing.data))
  binary.cols           <- intersect(all.features, binary.cols)
  numeric.cols          <- intersect(all.features, numeric.cols)
  cols.to.convert       <- intersect(all.features, factor.cols)


  ####### BALANCE THE TRAINING DATA ##############################################################
  set.seed(10)
  library(unbalanced)
  balancer.ubunder                <- ubUnder(X=curr.training.data[,-c(3)], Y=curr.training.data[,'DISCONT'], perc = balace.ratio, method = "percUnder")
  curr.training.data              <- cbind(balancer.ubunder$Y, balancer.ubunder$X)
  names(curr.training.data)[1]    <- paste("DISCONT")
  detach(package:unbalanced, unload=T, force=T) 
  detach(package:mlr, unload=T, force=T)
  
  
  ####### METADATA EDITOR ##############################################################
  for(i in 1:length(cols.to.convert))
  {
    cname                               <- cols.to.convert[i]
    # Convert to R Valid Names. Caret is very picky
    curr.testing.data[,cname]           <- as.factor(make.names(curr.testing.data[,cname])) 
    curr.training.data[,cname]          <- as.factor(make.names(curr.training.data[,cname]))
    table.for.validation[,cname]        <- as.factor(make.names(table.for.validation[,cname]))

    # Make sure that factor levels between training and testing match
    level.list <- union(levels(curr.training.data[,cname]), levels(curr.testing.data[,cname]))
    level.list <- union(levels(table.for.validation[,cname]),level.list)
    levels(curr.training.data[,cname])  <- level.list
    levels(curr.testing.data[,cname])   <- level.list
    levels(table.for.validation[,cname])   <- level.list
    
  }
  ##### FEATURE SELECTION ##############################################################
  ##### USING HALABI
  weights   <- random.forest.importance(DISCONT ~., curr.training.data[,setdiff(names(curr.training.data),c(halabi,"RPT","STUDYID"))], importance.type = 1)
  features  <- c(halabi, cutoff.k(weights,k))
  
  ##### UNIVARIATE (DEFAULT=DISABLED)
#   sub.fs      <- curr.training.data
#   sub.fs$RPT  <- NULL
#   weights     <- chi.squared(DISCONT ~., sub.fs)
#   features    <- cutoff.k(weights,k)

  ####### CLASSIFICATION / MODEL ##############################################################
  
  x                <- as.data.frame(curr.training.data[,features])
  y                <- as.factor(make.names(curr.training.data$DISCONT,unique = F))   
  set.seed(123)
  
  model.rf          <- randomForest(x=x, y=y, mtry=round(length(features)/rf.mtry), na.action = na.omit, probability=T, 
                                    ntree = (length(features)*rf.ntree),type="classification",replace = T)
  prob              <- predict(model.rf, curr.testing.data, type = "prob")[,2]
  val.prob          <- predict(model.rf, table.for.validation,type="prob")[,2]
  
  ####### OUTPUT ##############################################################
  OUTPUT.TABLE          <- cbind(curr.testing.data[,features],prob, as.factor(round(prob)), curr.testing.data$DISCONT)
  SCORING.TABLE         <- rbind(SCORING.TABLE, c(testing.name, dream9.score(prob, curr.testing.data$DISCONT)))
  ACC                   <- round(score_q2(prob, curr.testing.data$DISCONT)*100)
  
   setwd("~/Workspaces/git_repos/DREAM9.5/DATA/")
  ## WRITE OUTPUT AS CSV
  write.csv(OUTPUT.TABLE, file = paste("OUTPUT/SUBSETTEST-",testing.name,"-",ACC,".csv", sep=""))
  print(paste("---- FILE:", "SUBSETTEST-",testing.name,"-",ACC,".csv  WRITTEN TO HARDDRIVE ---",sep=""))
  write.csv(SCORING.TABLE, file = paste("OUTPUT/SCORE-",testing.name,"-",ACC,".csv", sep=""))
  print(paste("---- FILE:", "SCORE-",testing.name,"-",ACC,".csv WRITTEN TO HARDDRIVE ---",sep=""))
  
  
  ## WRITE VALIDATION OUTPUT AS CSV
  FINAL.TABLE             <- as.data.frame(cbind(as.character(table.for.validation$RPT),val.prob, as.numeric(round(val.prob))))
  colnames (FINAL.TABLE)  <- c("RPT","RISK","DISCONT")
  write.csv(FINAL.TABLE, file = paste("OUTPUT/VALIDATION-",testing.name,ACC,".csv", sep=""),row.names = FALSE)
  print(paste("---- FILE:", "VALIDATION-",ACC,".csv WRITTEN TO HARDDRIVE ---", sep=""))
  setwd(path.script)
  
} # END OF FOLD PER STUDY
} # END OF CROSS-FOLD VALIDATION


scoring.rows             <- SCORING.TABLE[,1]
SCORING.TABLE            <- SCORING.TABLE[,-1]
SCORING.TABLE            <- apply(SCORING.TABLE,2,as.numeric)
row.names(SCORING.TABLE) <- scoring.rows
print(SCORING.TABLE)
plot(cbind(CV,SCORING.TABLE[,1]),xlab=x.axis)
print(paste("MEAN AUC: ", mean(SCORING.TABLE[,1])))
CV[which.max(SCORING.TABLE[,1])]
write.csv(SCORING.TABLE, file = paste("OUTPUT/SCORE-",testing.name,"-",ACC,".csv", sep=""))