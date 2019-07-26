import org.apache.spark.ml.feature.Binarizer;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;


abstract class Classifier {

    private long FN,FP,TN,TP;
    private long totalTrue,totalFalse,trainTime,predictTime;
    private double accuracy,testError,precision,AUC;
    private Matrix confusion_matrix;

    public double getAccuracy() {
        return accuracy;
    }
    public double getPrecision() { return precision; }
    public Matrix getConfusionMatrix() { return confusion_matrix;  }
    public long getFN()           {  return FN;    }
    public long getTN()           {  return TN;    }
    public long getFP()           {  return FP;    }
    public long getTP()           {  return TP;    }
    public double getAUC()        {  return AUC;   }
    public double getTestError() {  return testError;    }
    public long getTotalFalse()  {  return totalFalse;   }
    public long getTotalTrue()   {  return totalTrue;    }
    public long getTrainTime()    {  return trainTime;     }
    public long getPredictTime()  {  return predictTime;   }

    public void setTrainTime(long t) { trainTime = t;}
    public void setPredictTime(long t) { predictTime = t;}

    public Dataset<Row> run(Dataset<Row> trainingData, Dataset<Row> testData){
        return null;
    }


    /* Calculate all metrics from prediction results */


    public void calculateMetrics(Dataset<Row> predictions){

        Binarizer binarizer = new Binarizer()
                .setInputCol("prediction")
                .setOutputCol("binarized_prediction")
                .setThreshold(0.5);

        Dataset<Row> predictionBinary = binarizer.transform(predictions);

        //Calculate all metrics
        Dataset<Row> correctPredictions = predictionBinary.where("label == prediction");
        this.totalTrue = correctPredictions.count();

        Dataset<Row> wrongPredictions = predictionBinary.where("label != prediction");
        this.totalFalse = wrongPredictions.count();

        Dataset<Row> countCorrect = correctPredictions.groupBy("label").count();
        Dataset<Row> countErrors = wrongPredictions.groupBy("label").count();

        this.TN = countCorrect.filter("label == 0.0").first().getLong(1);
        this.TP = countCorrect.filter("label == 1.0").first().getLong(1);
        this.FN = countErrors.filter("label == 0.0").first().getLong(1);
        this.FP = countErrors.filter("label == 1.0").first().getLong(1);

        BinaryClassificationMetrics binaryClassificationMetrics = new BinaryClassificationMetrics
                (predictions.select("label","prediction"));
        MulticlassMetrics multiclassMetrics = new MulticlassMetrics
                (predictions.select("prediction","label"));

        this.AUC = binaryClassificationMetrics.areaUnderROC();
        this.accuracy = multiclassMetrics.accuracy();
        this.testError = 1.0 - this.accuracy;
        this.confusion_matrix = multiclassMetrics.confusionMatrix();

    }
}

