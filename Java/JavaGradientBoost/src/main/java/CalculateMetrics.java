import org.apache.spark.ml.feature.Binarizer;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class CalculateMetrics {

   private long FN,FP,TN,TP;
   private long total_true,total_false;
   private double accuracy,test_error,precision,AUC;
   private Matrix confusion_matrix;

   public CalculateMetrics(){}

   public double getAccuracy() {
        return accuracy;
    }

   public double getPrecision() { return precision; }

   public Matrix getConfusion_matrix() { return confusion_matrix;  }

   public long getFN() {  return FN;    }

   public long getTN() {  return TN;    }

   public long getFP() {  return FP;    }

   public long getTP() {  return TP;    }

   public double getAUC() { return AUC; }

   public double getTest_error() {  return test_error;    }

   public long getTotal_false() {   return total_false;    }

   public long getTotal_true() {    return total_true;    }

   public void calculate(Dataset<Row> predictions) {

       Binarizer binarizer = new Binarizer().setInputCol("prediction").setOutputCol("binarized_prediction").setThreshold(0.5);
       Dataset<Row> predictionBinary = binarizer.transform(predictions);
       //Calculate all metrics
       Dataset<Row> correctPredictions = predictionBinary.where("label == prediction");
       this.total_true = correctPredictions.count();

       Dataset<Row> wrongPredictions = predictionBinary.where("label != prediction");
       this.total_false = wrongPredictions.count();

       Dataset<Row> countCorrect = correctPredictions.groupBy("label").count();
       Dataset<Row> countErrors = wrongPredictions.groupBy("label").count();

       this.TN = countCorrect.filter("label == 0.0").first().getLong(1);
       this.TP = countCorrect.filter("label == 1.0").first().getLong(1);
       this.FN = countErrors.filter("label == 0.0").first().getLong(1);
       this.FP = countErrors.filter("label == 1.0").first().getLong(1);

       BinaryClassificationMetrics binaryClassificationMetrics = new BinaryClassificationMetrics(predictions.select("label","prediction"));
       MulticlassMetrics multiclassMetrics = new MulticlassMetrics(predictions.select("prediction","label"));

       this.AUC = binaryClassificationMetrics.areaUnderROC();
       this.accuracy = multiclassMetrics.accuracy();
       this.test_error = 1.0 - this.accuracy;
       this.precision = multiclassMetrics.precision();
       this.confusion_matrix = multiclassMetrics.confusionMatrix();

   }

}
