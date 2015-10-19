                    DREAM CHALLENGES SUBMISSION (CHALLENGE 2.B)

                      TEAM YODA | UNIVERSITY OF WASHINGTON TACOMA
                                kayee@u.washington.edu
                                      SUMMER - 2015

Predicting Discontinuation of Docetaxel Treatment for Metastatic 
Castration-Resistant Prostate Cancer (mCRPC) with hill-climbing and Random Forest


1. INTRODUCTION
	Kevin Anderson, Seyed Khankhajeh, Daniel Kristiyanto, Kaiyuan Shi, Seth West contributed equally as first authors. 
	Kevin Anderson, Seyed Khankhajeh, Daniel Kristiyanto, Kaiyuan Shi, Seth West, Azu Lee, Qi Wei, Migao Wu, Yunhong Yin conducted empirical studies comparing the performance of various machine learning and feature selection algorithms on this dataset.  
	Daniel Kristiyanto served as the team captain and managed the submission uploads.  
	Ka Yee Yeung served as the PI, used this DREAM 9.5 challenge as a class project, provided guidance and co-ordinated this submission effort.
  

2. FILE LIST
	MAIN.R 				: Contains the main R script to analize and generate prediction.
	inc_functions.R 		: Contains functions such as data split, etc, required by Main Script.
	inc_datacleanup.R		: Contains function to perform data cleansing and pre-processes
	score.R				: Contains the scoring script to compute AUC as provided by Synapse
	CoreTable_training.csv 		: Contains the training data as provided by Synapse
	CoreTable_validation.csv	: Contains the Validation data as provided by Synapse.
	CoreTable_leaderboard.csv	: Contains the Validation data as provided by Synapse. This will be concatenated with CoreTable_validation.csv

	README.txt 			: this file.

3. REQUIRED R PACKAGES
	(FSelector) 
	(e1071)
	(timeROC)
	(caret)
	(unbalanced)
	(impute) By Bioconductor

4. VARIABLES
	CV 			: Numeric vector to perform thec cross-fold validation. Each folds takes about 25 minutes to run.  Sample: CV = 	c(1:10) 
	STUDY 			: Numeric vector ranged of 1 to 3; Script itterates through every study with following target (and use the other two as training data): 1. ACCENT2, 2. EFC6546, 3. CELGENE 4. ALL COMBINED. Sample: STUDY = c(1,2,3)

5. DATAFRAMES
Once run completed, these variables should be available:
	core_table_training     : Dataframe, Original core training data (Raw)
	core_table_validation	: Dataframe, Original core testing data (Raw)
	table.for.model      	: Dataframe, Cleaned and augmented core training data
	table.for.validation    : Dataframe, Cleaned and augmented core testing data
	curr.training.data    	: Dataframe, Subset of current fold training data
	curr.testing.data   	: Dataframe, Subset of current fold testing data
	SCORING.TABLE        	: Dataframe, Scoring results
	OUTPUT.TABLE          	: Dataframe, Subset of current testing data with predicted probability
	FINAL.TABLE           	: Dataframe, Subset of core testing data with predicted probability 

6. OUTPUT
Within each itteration, following files are generated:
SCORE-[STUDYNAME]-[AUC].csv		: Contains the scoring information.
SUBSETTEST-[STUDYNAME]-[AUC].csv	: Contains the Subset of the testing data used for the final prediction.
VALIDATION-[AUC].csv			: Contains the prediction of the validation data as required by Synapse.


