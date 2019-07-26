require("mlr")
require("gbm")

#project <- "/home/nyu/2018/summer/6513/vc1436/spark-projects/EmployeeAttrition/"
#setwd(project)

source("R/RGradientBoost/LoadAndPrepareData.R")
source("R/RGradientBoost/CalculateMetrics.R")

filePath <- "data/attrition_data.csv"
#load data
dataList <- loadAndPrepare(filePath)

trainData <- dataList$trainData
testData <- dataList$testData

task1 = makeClassifTask(data = trainData, target = "Attrition")
task2 = makeClassifTask(data = testData, target = "Attrition")

set.seed(43)

#make learner
lrn <- makeLearner("classif.gbm",predict.type="prob")

#train
mod <- train(lrn,task=task1)

#predict
predictions <- predict(mod,task=task2)

#calculate metrics
result <- calculateMetrics(predictions)

#print results
print(paste0("================================="))
print(paste0("Count of training data: " , nrow(trainData)))
print(paste0("Count of test data: " , nrow(testData)))
print(paste0("Total True Predictions: " , result$total_true))
print(paste0("Total False Predictions: ", result$total_true))
print(paste0("================================="))
print(paste0("Area under ROC: " , result$performance$auc))
print(paste0("Accuracy: " ,result$performance$acc))
print(paste0("Test Error: ",(1-result$performance$acc)))
confusion_matrix = as.data.frame(result$confusionMatrix$result,row.names = F,colname=F)
tp = confusion_matrix[2,2]
fp = confusion_matrix[2,1]
tn = confusion_matrix[1,1]
fn = confusion_matrix[1,2]  
print(paste0("TP: " ,tp," FP: " ,fp ," FN: ",fn ," TN: ",tn))
print(paste0("================================="))
print(paste0("Confusion matrix: " ))
cat("\t\t Actual\nPrediction\t 0\t 1\n")
cat("\t0\t",tn,"\t" , fp,"\n")
cat("\t1\t",fn,"\t" , tp,"\n")
print(paste0("================================="))
rm(list=ls())
