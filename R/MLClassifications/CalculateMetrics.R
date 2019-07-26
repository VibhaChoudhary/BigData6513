calCulateMetrics <- function(predictions,mod){
  data <- predictions$data
  total_true <- nrow(data[data$truth == data$response,])
  total_false <- nrow(data[data$truth != data$response,])
  confusionMatrix <- calculateConfusionMatrix(predictions)
  mets <- list(timetrain,timepredict,acc,auc,tp,tn,fp,fn,tpr,fpr)
  performance <- as.list(performance(predictions,mets,model=mod))
  list(confusionMatrix=confusionMatrix,performance=performance,total_false=total_false,total_true=total_true)
}