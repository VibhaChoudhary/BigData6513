import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class GradientBoost extends Classifier {
    public GradientBoost() {

    }

    @Override
    public Dataset<Row> run(Dataset<Row> trainingData, Dataset<Row> testData) {
        GBTClassifier gbtClassifier = new GBTClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features").setMaxDepth(5).setMaxIter(100).setSeed(43);
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{gbtClassifier});

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