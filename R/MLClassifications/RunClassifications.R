require("mlr")
require("gbm")
require("doParallel")
filePath <- "data/attrition_data.csv"

source("R/MLClassifications/LoadAndPrepareData.R")
source("R/MLClassifications/CalculateMetrics.R")

registerDoParallel(detectCores())
filePath <- "data/attrition_data.csv"
#load data
dataList <- loadAndPrepare(filePath)

trainData <- dataList$trainData
testData <- dataList$testData


print("=================================")
print(paste0("Records in train data: " , nrow(trainData)))
print(paste0("Records in test data: "  , nrow(testData)))
print("=================================")

task1 = makeClassifTask(data = trainData, target = "Attrition")
task2 = makeClassifTask(data = testData, target = "Attrition")

set.seed(43)

learners <- list(
  "Gradient Boost" = "classif.gbm",
  "Decision Tree" = "classif.rpart",
  "Logistic Regression" = "classif.logreg",
  "random Forest"= "classif.randomForest", 
  "Naive Bayes" = "classif.naiveBayes"
  )

result <- foreach(i = 1:length(learners),
                  .combine=c,
                  .packages=c("mlr","gbm")) %dopar% {
  
  lrn <- makeLearner(learners[[i]],predict.type="prob")
        
  #train
  mod <- train(lrn,task=task1)
  #predict
  predictions <- predict(mod,task=task2)
  #calculate metrics
  result <- calCulateMetrics(predictions,mod)
  id <- names(learners[i])
  cm <- as.data.frame(result$confusionMatrix$result[c(1,2),c(1,2)])
  list(classifier = id,measures = result)
  }

result_df <- foreach(i=1:length(learners),.combine=rbind) %dopar%{
   j <- i-1; 
   id <- result[i+j]$classifier
   measures <- result[i+j+1]$measures
   cm <- measures$confusionMatrix$result
   tp = cm[2,2]
   fp = cm[2,1]
   tn = cm[1,1]
   fn = cm[1,2]
   performance <- measures$performance
   total_true<- measures$total_true
   total_false<- measures$total_false
   list(
     "Classifier"=id,
     "True"=total_true,
     "False"=total_false,
     "Train_time"=performance$timetrain,
     "Predict_time"=performance$timepredict,
     "Acc"=performance$acc,
     "AUC"=performance$auc,
     "TP"=tp,
     "TN"=tn,
     "FP"=fp,
     "FN"=fn
   )
   
}

result_df <- as.data.table(result_df,row.names=F)
print(result_df)

