calculateMetrics <- function(predictions){
  data <- predictions$data
  total_true <- nrow(data[data$truth == data$response,])
  total_false <- nrow(data[data$truth != data$response,])
  confusionMatrix <- calculateConfusionMatrix(predictions)
  mets <- list(timetrain,timepredict,acc,auc,tp,tn,fp,fn)
  performance <- as.list(performance(predictions,model=mod,measures=mets))
  list(confusionMatrix=confusionMatrix,performance=performance,total_false=total_false,total_true=total_true)
}