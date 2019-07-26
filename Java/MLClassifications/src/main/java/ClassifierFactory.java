public class ClassifierFactory {

    public Classifier getClassifier(String classifierType){
        if(classifierType == null){
            return null;
        }
        if(classifierType.equalsIgnoreCase("DST")) return new DecisionTree();
        else if(classifierType.equalsIgnoreCase("GBT")) return new GradientBoost();
        else if(classifierType.equalsIgnoreCase("RF")) return new RandomForest();
        else if(classifierType.equalsIgnoreCase("NB")) return new NaiveBayesC();
        else if(classifierType.equalsIgnoreCase("LR")) return new LogisticRegressionC();
        return null;
    }
}

