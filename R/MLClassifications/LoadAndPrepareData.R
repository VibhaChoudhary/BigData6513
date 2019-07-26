require("data.table")
require("dplyr")
require("caTools")

loadAndPrepare = function(filepath){
  data <- fread(file = filepath, sep=",",header=T)
  data_f <- data %>% mutate_if(sapply(data,is.character),as.factor)
  data_c <- data_f %>% mutate_if(sapply(data_f,is.factor),as.numeric)
  data_c <- data_c %>% mutate(Attrition = as.numeric(Attrition)-1)
  data_c <- data_c %>% mutate(Attrition = as.factor(Attrition))
  set.seed(43)
  sample <- sample.split(data_c, SplitRatio = 0.70)
  trainData <- subset(data_c, sample == TRUE)
  testData <- subset(data_c, sample == FALSE)
  list(trainData=trainData,testData=testData)
}
