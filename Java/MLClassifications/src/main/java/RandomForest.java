import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class RandomForest extends Classifier {
    public RandomForest(){

    }

    @Override
    public Dataset<Row> run(Dataset<Row> trainingData, Dataset<Row> testData) {
        RandomForestClassifier randomForestClassifier = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features").setSeed(43).setMaxDepth(5);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{randomForestClassifier});

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
