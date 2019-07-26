import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class EmployeeAttrition {

   public static void main(String args[]){
       Logger.getLogger("org").setLevel(org.apache.log4j.Level.OFF);

       //Create a spark session
       SparkSession spark = SparkSession
               .builder()
               .appName("DTEmployeeAttrition")
               .config("spark.master", "local")
               .getOrCreate();

       //Load the data in spark session
       LoadAndPrepareData loadAndPrepareData = new LoadAndPrepareData();

       //Get training and test data containing label and features
       Dataset<Row> splits[] = loadAndPrepareData.load(spark);
       Dataset<Row> trainData = splits[0];
       Dataset<Row> testData = splits[1];

       //Set up the classifier
       LogisticRegression classifier = new LogisticRegression()
               .setLabelCol("label")
               .setFeaturesCol("features");

       //Create a pipeline of tasks
       Pipeline pipeline = new Pipeline()
               .setStages(new PipelineStage[]{classifier});

       // Train model.
       PipelineModel model = pipeline.fit(trainData);

       //Make Prediction
       Dataset<Row> predictions  = model.transform(testData);

       //Calculate all metrics
       CalculateMetrics calculateMetrics = new CalculateMetrics();
       calculateMetrics.calculate(predictions);

       //Print metrics
       System.out.println("=================================");
       System.out.println("Count of training data: " + trainData.count());
       System.out.println("Count of test data: " + testData.count());
       System.out.println("=================================");
       System.out.println("Total True Predictions: " + calculateMetrics.getTotal_true());
       System.out.println("Total False Predictions: " + calculateMetrics.getTotal_false());
       System.out.println("=================================");
       System.out.println("Area under ROC: " + calculateMetrics.getAUC());
       System.out.println("Accuracy: " + calculateMetrics.getAccuracy());
       System.out.println("Test Error: " + calculateMetrics.getTest_error());
       System.out.println("=================================");
       System.out.println("TP: " + calculateMetrics.getTP() +" FP: " + calculateMetrics.getFP() +
       " FN: " + calculateMetrics.getFN() +" TN: " + calculateMetrics.getTN());
       System.out.println("=================================");
       System.out.println("Confusion matrix: " );
       System.out.println("\t\t  Actual\nPrediction\t0\t1" );
       double[] confusion_matrix = calculateMetrics.getConfusion_matrix().toArray();
       System.out.println("\t0\t"+confusion_matrix[0]+"\t" + confusion_matrix[1]);
       System.out.println("\t1\t"+confusion_matrix[2]+"\t" + confusion_matrix[3]);
       System.out.println("=================================");
   }

}
