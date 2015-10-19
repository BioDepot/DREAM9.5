#' ###############################################
#'                  DATA CLEAN UP
#' 
#' 
#' 
#' Take a DREAM 9.5 data, and return with cleaned up version of it.
#' Args: x is Dream 9.5 Core Table spesific dataframe.
#' Return: Clean Dataframe
#'
#' Data cleansing including:
#'  - Remove uneeded features:
#'      - Reason: Single Values = (DOMAIN, LKADT_PER, HGTBLCAT, WGTBLCAT, TRT1_ID)
#'      - Reason: Completely Empty = (HEAD_AND_NECK, STOMACH, PANCREAS, THYROID)
#'      - Reason: Missing in more than 2 studies = (TESTO, RBC, LYM)
#'      - Reason: Redundant = (AGEGRP)
#'  - Leaving Dependant Variables (DEATH, ENDTRS_C, ENTRT_PC, LKADT_P, PER_REF, LKADT_REF) and Target (DISCONT) intact
#'  - Impute missing values with KNN or Mean
#'  - Clean up ".", NAs, Y and Yes into binary values.
#'
#'  @param x Dream 9 dataframe
#'  @return Clean dataframe. It may contain less features, but the no of rows should be the same with it's input.
#'  



dream9.cleanData <- function(x)
{
  require(impute)
  
  
  if (class(x) != 'data.frame')
  {
    cat("Please provide a data frame!");
    return();
  }
  
  core.table <- x
  core.table$DISCONT <- as.character(core.table$DISCONT)
  core.table$DISCONT[!(core.table$DISCONT %in% c(0,1))] <- NA
  
  
  ## DELETE ROWS THAT HAVE . AS DISCONT
  # core.table <- core.table[!(core.table$DISCONT == '.') ,]
  
  # REMOVE THE UNEEDED COLUMNS ####
  ## SINGLE VALUE        (DOMAIN, LKADT_PER, HGTBLCAT, WGTBLCAT, TRT1_ID)
  ## COMPLETELY EMPTY    (HEAD_AND_NECK, STOMACH, PANCREAS, THYROID)
  ## DEPENDANT VARIABLES (DEATH, ENDTRS_C, ENTRT_PC, LKADT_P, PER_REF, LKADT_REF)
  ## 2 STUDY MISSING     (TESTO, RBC, LYM)
  ## REDUNDANT           (AGEGRP)
  
  cat("Cleaning/Removing Columns...")
  remove.col.names <- c("DOMAIN", "LKADT_PER", "HGTBLCAT", "WGTBLCAT", "TRT1_ID"
                        , "HEAD_AND_NECK", "STOMACH", "PANCREAS", "THYROID"
                        , "DEATH", "ENDTRS_C", "ENTRT_PC", "LKADT_P", "PER_REF", "LKADT_REF"
                        , "TESTO", "RBC", "LYM",'AGEGRP')
  
  dependant.cols   <- c("DEATH", "ENDTRS_C", "ENTRT_PC", "LKADT_P", "PER_REF", "LKADT_REF")
  ## LEAVE DEPENDANT VARIABLES INTACT
  remove.col.names <- setdiff(remove.col.names, dependant.cols)
  
  remove.col.ids   <- which(colnames(core.table) %in% remove.col.names)
  core.table       <- core.table[ ,-(remove.col.ids)]
  
  
  # GLEAS_DX IS NUMERIC (BUT CATEGORICAL, SO "." CAN BE NA)
  ## IF GLEAS = ZERO COMMENT OUT THIS LINE
  core.table$GLEAS_DX[core.table$GLEAS_DX == "."] <- NA
  core.table$TSTAG_DX[core.table$TSTAG_DX == "."] <- NA
  
  ## LEAVE THIS LINE REGARDLESS
  core.table$GLEAS_DX <- as.numeric(core.table$GLEAS_DX)
  
  ## IF GLEAS = ZERO COMMENT OUT THIS LINE
  # core.table$GLEAS_DX[is.na(core.table$GLEAS_DX)] <- round(mean(core.table$GLEAS_DX[core.table$GLEAS_DX], na.rm=T)) ## IF GLEAS = ZERO COMMENT OUT THIS LINE
  
  # (DK) NO, GLEAS_DX	& TSTAG_DX NOT NUMBER
  core.table$GLEAS_DX[is.na(core.table$GLEAS_DX)] <- 'UNKNOWN'
  core.table$TSTAG_DX[is.na(core.table$TSTAG_DX)] <- 'UNKNOWN'
  
  ##(DK) IF REGION IS ASIA.PACIFIC OR AFRICA CHANGE IT TO 'OTHER
  core.table$REGION_C <- as.character(core.table$REGION_C)
  core.table$REGION_C[core.table$REGION_C == "ASIA/PACIFIC"] <- 'OTHER'
  core.table$REGION_C[core.table$REGION_C == "AFRICA"] <- 'OTHER'
  core.table$REGION_C[is.na(core.table$REGION_C)] <- 'UNKNOWN'
  
  # FIX AGEGRP, MAKE NUMERIC AND REMOVE THE >= 85 VALUE ## (DK) LEAVE AGE ALONE.
  #   core.table$AGEGRP[core.table$AGEGRP == ">=85"] <- 85
  #   core.table$AGEGRP <- as.numeric(core.table$AGEGRP)
  #
  # MAKE SMOKING Yes, No and Missing
  core.table$SMOKE[!(core.table$SMOKE %in% c('YES','NO'))] <- 'UNKNOWN'
  
  ## CONVERT THE .'S AND BLANKS TO NA
  blanks.col.names <- c("DISCONT","SMOKFREQ","SMOKSTAT","ECOG_C","TSTAG_DX","TRT2_ID","TRT3_ID", "BMI","HEIGHTBL","WEIGHTBL")
  blanks.col.ids   <- which(colnames(core.table) %in% blanks.col.names)
  
  for(c in blanks.col.ids)
  {
    core.table[,c][core.table[,c] %in% c(".","")] <- NA
  }
  
  
  core.table$TRT2_ID[is.na(core.table$TRT2_ID)]     <- 'UNKNOWN'
  core.table$TRT3_ID[is.na(core.table$TRT3_ID)]     <- 'UNKNOWN'
  core.table$SMOKFREQ[is.na(core.table$SMOKFREQ)]   <- 'UNKNOWN'
  core.table$SMOKSTAT[is.na(core.table$SMOKSTAT)]   <- 'UNKNOWN'
  core.table$TSTAG_DX[is.na(core.table$TSTAG_DX)]   <- 'UNKNOWN'
  core.table$ECOG_C[is.na(core.table$ECOG_C)]       <- 'UNKNOWN' #Added
  
  ## REMOVE SPACES AND SPECIAL CHARACTERS FROM STRING VALUES.
  core.table$REGION_C <- make.names(core.table$REGION_C, unique = F)
  core.table$SMOKFREQ <- make.names(core.table$SMOKFREQ, unique = F)
  core.table$SMOKSTAT <- make.names(core.table$SMOKSTAT, unique = F)
  core.table$AGEGRP2  <- make.names(core.table$AGEGRP2,  unique = F)
  
  # CONVERT THE BINARY COLS TO 1,0 ####
  yes_no.col.names <- c("NON_TARGET","TARGET","BONE","RECTAL","LYMPH_NODES","KIDNEYS","LUNGS"
                        ,"LIVER","PLEURA","OTHER","PROSTATE","ADRENAL","BLADDER","PERITONEUM"
                        ,"COLON","SOFT_TISSUE"
                        ,"ABDOMINAL","ORCHIDECTOMY","PROSTATECTOMY","TURP","LYMPHADENECTOMY"
                        ,"SPINAL_CORD_SURGERY","BILATERAL_ORCHIDECTOMY","PRIOR_RADIOTHERAPY"
                        ,"ANALGESICS","ANTI_ANDROGENS","GLUCOCORTICOID","GONADOTROPIN"
                        ,"BISPHOSPHONATE","CORTICOSTEROID","IMIDAZOLE","ACE_INHIBITORS"
                        ,"BETA_BLOCKING","HMG_COA_REDUCT","ESTROGENS","ANTI_ESTROGENS"
                        ,"ARTTHROM","CEREBACC","CHF","DVT","DIAB","GASTREFL","GIBLEED"
                        ,"MI","PUD","PULMEMB","PATHFRAC","SPINCOMP","COPD","MHBLOOD","MHCARD"
                        ,"MHCONGEN","MHEAR","MHENDO","MHEYE","MHGASTRO","MHGEN","MHHEPATO"
                        ,"MHIMMUNE","MHINFECT","MHINJURY","MHINVEST","MHMETAB","MHMUSCLE"
                        ,"MHNEOPLA","MHNERV","MHPSYCH","MHRENAL","MHRESP","MHSKIN","MHSOCIAL"
                        ,"MHSURG","MHVASC")
  
  yes_no.col.ids <- which(colnames(core.table) %in% yes_no.col.names)
  
  for(c in yes_no.col.ids)
  {
    # make NAs = 0
    core.table[,c] <- as.character(core.table[,c])
    core.table[is.na(core.table[ ,c]), c] <- 0
    
    yes.rows <- toupper(core.table[ ,c]) %in% c('Y','YES')
    
    if(sum(yes.rows, na.rm = T) > 0)
    {
      core.table[yes.rows,c] <- 1
    }
    
    no.rows <- (core.table[,c] != 1)
    
    if(sum(no.rows, na.rm = T  ) > 0)
    {
      core.table[no.rows,c]  <- 0
    }
    
    core.table[,c]         <- factor((core.table)[,c], levels=c("0","1")) #Covert it as factor
  }
  
  # ADD A NEW COLUMN OF SUMS # (DK) DISABLE IT FOR NOW TO REDUCE NOISES
  #   canc.col.names <- c('BONE','RECTAL','LYMPH_NODES','KIDNEYS','LUNGS','LIVER','PLEURA','OTHER','PROSTATE','ADRENAL',
  #   'BLADDER','PERITONEUM','COLON','HEAD_AND_NECK','SOFT_TISSUE','STOMACH','PANCREAS','THYROID','ABDOMINAL')
  #
  #   canc.col.ids   <- which(colnames(core.table) %in% canc.col.names)
  #   core.table$CancerSums <- as.factor(rowSums(sapply(core.table[ , canc.col.ids], as.numeric)-1))
  #
  #   med.col.names <- c('MHBLOOD',	'MHCARD',	'MHCONGEN',	'MHEAR',	'MHENDO',	'MHEYE',	'MHGASTRO',	'MHGEN',	'MHHEPATO',
  #                      'MHIMMUNE',	'MHINFECT',	'MHINJURY',	'MHINVEST',	'MHMETAB',	'MHMUSCLE',	'MHNEOPLA',	'MHNERV',	'MHPSYCH',
  #                      'MHRENAL',	'MHRESP',	'MHSKIN',	'MHSOCIAL',	'MHSURG',	'MHVASC')
  #
  #   med.col.ids   <- which(colnames(core.table) %in% med.col.names)
  #   core.table$MedBodySums <- as.factor(rowSums(sapply(core.table[ , med.col.ids], as.numeric)-1))
  #
  
  # GET INDEXES FOR THE PURELY NUMERIC COLUMNS
  start.ind <- which(names(core.table) == "TRT3_ID") + 1
  end.ind   <- which(names(core.table) == "NON_TARGET") - 1
  num.cols  <- start.ind:end.ind
  
  # SET NUMERIC COLUMNS 21:44 TO NUMERIC
  for (i in num.cols)
  {
    core.table[,i] <- as.numeric(core.table[,i])
  }
  
  cat("SUCCESS\n")
  
  # DATA IMPUTATION PROCESS
  cat("Cleaning by Study...")
  core.table <- impute_by_study(core.table)
  
  ## Set Columns levels (DK)
  factorcols <- c('STUDYID',	'RPT',	'DISCONT',	'GLEAS_DX',	'TSTAG_DX',	'RACE_C',	'REGION_C',	'SMOKE',	'SMOKFREQ',
                  'SMOKSTAT',	'ECOG_C',	'TRT2_ID',	'TRT3_ID',	'NON_TARGET',	'TARGET',	'BONE',	'RECTAL',	'LYMPH_NODES',
                  'KIDNEYS',	'LUNGS',	'LIVER',	'PLEURA',	'OTHER',	'PROSTATE',	'ADRENAL',	'BLADDER',	'PERITONEUM',
                  'COLON',	'SOFT_TISSUE',	'ABDOMINAL',	'ORCHIDECTOMY',	'PROSTATECTOMY',	'TURP',	'LYMPHADENECTOMY',
                  'SPINAL_CORD_SURGERY',	'BILATERAL_ORCHIDECTOMY',	'PRIOR_RADIOTHERAPY',	'ANALGESICS',	'ANTI_ANDROGENS',
                  'GLUCOCORTICOID',	'GONADOTROPIN',	'BISPHOSPHONATE',	'CORTICOSTEROID',	'IMIDAZOLE',	'ACE_INHIBITORS',
                  'BETA_BLOCKING',	'HMG_COA_REDUCT',	'ESTROGENS',	'ANTI_ESTROGENS',	'ARTTHROM',	'CEREBACC',	'CHF',
                  'DVT',	'DIAB',	'GASTREFL',	'GIBLEED',	'MI',	'PUD',	'PULMEMB',	'PATHFRAC',	'SPINCOMP',	'COPD',
                  'MHBLOOD',	'MHCARD',	'MHCONGEN',	'MHEAR',	'MHENDO',	'MHEYE',	'MHGASTRO',	'MHGEN',	'MHHEPATO',
                  'MHIMMUNE',	'MHINFECT',	'MHINJURY',	'MHINVEST',	'MHMETAB',	'MHMUSCLE',	'MHNEOPLA',	'MHNERV',	'MHPSYCH',
                  'MHRENAL',	'MHRESP',	'MHSKIN',	'MHSOCIAL',	'MHSURG',	'MHVASC', 'AGEGRP2')
  for(i in 1:length(factorcols))
  {
    curr.colname <- factorcols[i]
    if(curr.colname %in% colnames(core.table))
    {
      #return.data[,factorcols]      <- lapply(return.data[,factorcols], factor)
      core.table[,curr.colname] <- as.factor(core.table[,curr.colname])
    }
  }
  cat("SUCCESS\n")
  cat("Cleanup complete, table returned.")
  
  return(core.table)
  
}



impute_by_study <- function(x = NULL)
{
  
  
  if(class(x) != "data.frame")
  {
    cat("Please provide a dataframe");
    return()
  }
  
  
  # SPLIT THE DATA INTO THREE TABLES: ASCENT2, CELGENE, EFC
  
  study.split <- x$STUDYID
  split.table <- split(x, study.split)
  
  study.count <- length(split.table)
  
  col.names <- colnames(x)
  
  for (i in 1:study.count)
  {
    
    # GET THE STARTING AND ENDING COL INDEXES OF THE DATA WE ARE IMPUTITING.
    start.ind <- which(names(x) == "TRT3_ID") + 1
    end.ind <- which(names(x) == "NON_TARGET") - 1
    num.cols <- start.ind:end.ind
    
    # CAPTURE THE CURRENT STUDY AS THE CURRENT TABLE
    curr.table <- split.table[[i]]
    
    ## REMOVE ROWS AND COLS WITH HIGH MISSING DATA ####
    
    ## REMOVES ROWS WITH 50% MISSING DATA
    #     row.missing <- apply(curr.table[, num.cols], 1, function(z) { sum(is.na(z))})
    #     remove.row.names <- names(row.missing)[which((row.missing / ncol(curr.table[,num.cols])) > .50)]
    #     remove.row.ind <- which(rownames(curr.table) %in% remove.row.names)
    #     if(length(remove.row.ind) > 0) {
    #       curr.table <- curr.table[-remove.row.ind, ]
    #     }
    
    ## REMOVES COLS WITH 79% MISSING DATA
    col.missing <- apply(curr.table[, num.cols], 2, function(z) { sum(is.na(z)) / length(is.na(z))})
    remove.col.names <- names(col.missing)[which((col.missing) > .79)]
    remove.col.ind   <- which(names(curr.table) %in% remove.col.names)
    
    if (length(remove.col.ind) > 0)
    {
      curr.table <- curr.table[, -remove.col.ind]
    }
    
    # RE-CALCULATE THE NUMERIC COLUMNS
    start.ind <- which(names(curr.table) == "TRT3_ID") + 1
    end.ind <- which(names(curr.table) == "NON_TARGET") - 1
    num.cols <- start.ind:end.ind
    
    
    ### IMPUTE SECTION ####
    curr.table.knn <- as.matrix(curr.table[, num.cols])
    curr.impute <- impute.knn (curr.table.knn, k=40)
    
    curr.table[,num.cols] <- curr.impute$data
    
    
    ### _MEAN ####
    mean.col.names <- c("BMI","HEIGHTBL","WEIGHTBL")
    mean.col.ids <- which(colnames(curr.table) %in% mean.col.names)
    
    ## SET MISSING VALUES TO THE MEAN
    for (c in mean.col.ids)
    {
      curr.table[,c] <- as.numeric(curr.table[,c])
      curr.table[,c][is.na(curr.table[,c])] <- mean(curr.table[,c], na.rm = T)
    }
    
    
    # WRITE OUT THE CLEANED UP CORE TABLE ####
    #    filename <- paste(names(split.table[i]),"_knn.csv", sep="")
    #    write.csv(curr.table, file=filename, row.names = FALSE)
    
    # SAVE THE CHANGES BACK TO THE OBJECT
    split.table[[i]] <- curr.table
  }
  
  
  #### NOW I MUST SIMPLY DEVISE A BETTER WAY OF PUTTING THESE STUDIES BACK TOGETHER DYNAMICALLY
  for(curr.col in col.names)  ## FOR EACH COLUMN IN THE ORIGINAL DATASET
  {
    curr.col.data <- NULL
    for (i in 1:study.count)  ## LOOK THROUGH EACH STUDY SPLIT
    {
      # GET THE NEEDED COLUMN INDEX
      curr.ind <- which(colnames(split.table[[i]]) == curr.col)
      
      # IF I HAVE AN INDEX, APPEND THAT COLUMN TO THE CURRENT DATA COLUMN, OTHERWISE USE NULL
      if (!length(curr.ind) == 0) ## I HAVE AN INDEX
      {
        if (class(split.table[[i]][,curr.ind]) == "factor")   #special case for concatenating factos
        {
          curr.col.data <- c(curr.col.data, as.character(split.table[[i]][,curr.ind]))
        } else {
          curr.col.data <- c(curr.col.data, split.table[[i]][,curr.ind ])
        }
      } else {  # WE NEED TO APPEND A BUNCH OF NAS
        if (class(curr.col.data) == "factor")
        {
          curr.col.data <- c(as.character(curr.col.data), rep(NA, nrow(split.table[[i]])))
        } else {
          curr.col.data <- c(curr.col.data, rep(NA, nrow(split.table[[i]])))
        }
      }
    }
    
    # ADD THE COLUMN AND GO TO THE NEXT ONE
    if (curr.col == "STUDYID")   ## for the first loop build a dataframe.  for the rest, just append to it
    {
      return.data <- data.frame(STUDYID=curr.col.data,  stringsAsFactors=FALSE)
    } else {
      return.data[paste(curr.col)] <- curr.col.data
    }
  }
  
  # DELETE ANY COLUMNS THAT ARE ALL NAs
  return.data <- return.data[ , colSums(is.na(return.data)) < nrow(return.data)]
  
  cat("Imputing remaining missing values...")
  
  #### MEDIAN ROWS  MG and TPRO!
  if ("MG" %in% colnames(return.data))
  {
    return.data[is.na(return.data$MG),"MG"]     <- median(return.data$MG)
  }
  
  if ("TPRO" %in% colnames(return.data))
  {
    return.data[is.na(return.data$TPRO),"TPRO"] <- median(return.data$TPRO)
  }
  
  ### MEAN ROWS NA.
  if ("NA." %in% colnames(return.data))
  {
    return.data[is.na(return.data$NA.),"NA."] <- median(return.data$NA.)
  }
  
  ### DO KNN FOR THE REST
  start.ind <- which(names(return.data) == "TRT3_ID") + 1
  end.ind <- which(names(return.data) == "NON_TARGET") - 1
  num.cols <- start.ind:end.ind
  
  ## REMOVE COLUMNS WITH > 79 % MISSING DATA
  col.missing      <- apply(return.data[, num.cols], 2, function(z) { sum(is.na(z)) / length(is.na(z))})
  remove.col.names <- names(col.missing)[which((col.missing) > .79)]
  remove.col.ind   <- which(names(return.data) %in% remove.col.names)
  
  if (length(remove.col.ind) > 0)
  {
    return.data <- return.data[, -remove.col.ind]
  }
  
  ## RE-GENERATE THE COLNUM INDEXES
  start.ind <- which(names(return.data) == "TRT3_ID") + 1
  end.ind <- which(names(return.data) == "NON_TARGET") - 1
  num.cols <- start.ind:end.ind
  
  return.data.knn  <- as.matrix(return.data[, num.cols])
  clean.impute <- impute.knn (return.data.knn, k=40)
  
  return.data[,num.cols] <- clean.impute$data
  
  cat("SUCCESS\n")
  
  
  return(return.data)
  
}


#'  Dream 9.5 SCORING
#'  @param pred Prediction Probability
#'  @param y True Value
#'  @return A Vector list of of Score containing (in order) ("AUC","ACC","F1", "PREC", "REC","TP","FP","TN","FN")
#'  @export


## SCORING FUNCTION
## REQUIRES INCLUDE SCORE R FROM SYNAPSE
dream9.score <- function(pred, y)
{
  #   install.packages("survival")
  #   library("survival")
  #   require("survival")
  source("score.R")
  TN <- sum((round(pred) == 0) & (y == 0))
  TP <- sum((round(pred) == 1) & (y == 1))
  FP <- sum((round(pred) == 1) & (y == 0))
  FN <- sum((round(pred) == 0) & (y == 1))
  prec <- TP / (TP + FP)
  rec  <- TP / (TP + FN)
  
  F1               <- (2 * TP) / ((2*TP) + FP + FN)
  acc              <- (TN + TP)  / (TN + TP + FP + FN)
  auc              <- score_q2(pred, y)
  CF.Matrix        <- c(auc,acc,F1, prec, rec, as.numeric(TP),as.numeric(FP),as.numeric(TN),as.numeric(FN))
  names(CF.Matrix) <- c("AUC","ACC","F1", "PREC", "REC","TP","FP","TN","FN")
  print(CF.Matrix)
  return(CF.Matrix)
}


#'  Augment.
#'  @param x Dream 9 dataframe
#'  @param col ONE column/variable name to explode
#'  @return Dataframe with augmented variables.
#'  @export

dream9.explode <- function(x, col)
{
  if (class(x) != 'data.frame')
  {
    cat("Please provide a data frame!");
    return();
  }
  
  if(!(col %in% colnames(x)))
  {
    print("Variable not exist.")
    return()
  }
  else
  {
    tmp       <- as.data.frame(x)
    tmp[,col] <- as.factor(tmp[,col])
    f         <- paste("~0+",col)
    tmp       <- model.matrix(as.formula(f),data=tmp)
    tmp       <- apply(tmp,2,factor)
    x         <- cbind(x, tmp)
    print(paste("New variable added:", colnames(tmp)))
    x         <- x[,!(colnames(x)==col)]
    print(paste("Variable removed:",col))
    return(x)
  }
}



