import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class NaiveBayesC extends Classifier{
    public NaiveBayesC(){

    }
     @Override
     public Dataset<Row> run(Dataset<Row> trainingData, Dataset<Row> testData) {
            NaiveBayes naiveBayes = new NaiveBayes()
                    .setLabelCol("label")
                    .setFeaturesCol("features");


            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[]{naiveBayes});
         long start = System.currentTimeMillis();

         // Train model. This also runs the indexers.
         PipelineModel model = pipeline.fit(trainingData);

         long stop = System.currentTimeMillis();
         this.setTrainTime((stop - start)/1000);
         start = System.currentTimeMillis();
         // Make predictions.
         Dataset<Row> predictions = model.transform(testData);
         stop = System.currentTimeMillis();
         this.setPredictTime((stop - start)/1000);
         return predictions;
     }
}

