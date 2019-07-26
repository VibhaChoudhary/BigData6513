import org.apache.log4j.LogManager;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import javax.swing.JFileChooser;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class RunClassification {
    public static void main(String []args){
        
            String path="data/attrition_data.csv";
        
            long start = System.currentTimeMillis();
            LogManager.getLogger("org").setLevel(org.apache.log4j.Level.OFF);

            SparkSession spark = SparkSession
                    .builder()
                    .appName("MLClassifications")
                    //.config("spark.master", "local")
                    .getOrCreate();
            /* load data */
            LoadAndPrepareData loadAndPrepareData = new LoadAndPrepareData();
            Dataset<Row> splits[] = loadAndPrepareData.load(spark,path);

            Dataset<Row> trainData = splits[0];
            Dataset<Row> testData = splits[1];

            /* run classifiers */
            ClassifierFactory classifierFactory = new ClassifierFactory();

            Classifier gbtClassifier = classifierFactory.getClassifier("GBT");
            Classifier dstClassifier =  classifierFactory.getClassifier("DST");
            Classifier nbClassifier  = classifierFactory.getClassifier("NB");
            Classifier lrClassifier = classifierFactory.getClassifier("LR");
            Classifier rfClassifier = classifierFactory.getClassifier("RF");

            List<Classifier> classifiers = Arrays.asList(gbtClassifier,dstClassifier,nbClassifier,lrClassifier,rfClassifier);
            classifiers.parallelStream().forEach(
                    classifier -> classifier.calculateMetrics(classifier.run(trainData,testData))
            );

            List<String> columnNames = Arrays.asList("Classifier","Total_true","Total_false","Train_time(s)","Predict_time(s)","TP","TN","FP","FN","ACC","AUC");
            List<Object> data = new ArrayList<Object>();
            data.add(Arrays.asList("GBT",gbtClassifier.getTotalTrue(),gbtClassifier.getTotalFalse(),
                            gbtClassifier.getTrainTime(), gbtClassifier.getPredictTime(),
                            gbtClassifier.getTP(), gbtClassifier.getTN(), gbtClassifier.getFP(), gbtClassifier.getFN(),
                            gbtClassifier.getAccuracy(), gbtClassifier.getAUC()));

            data.add(Arrays.asList("DST",dstClassifier.getTotalTrue(),dstClassifier.getTotalFalse(),
                            dstClassifier.getTrainTime(),dstClassifier.getPredictTime(),
                            dstClassifier.getTP(),dstClassifier.getTN(),dstClassifier.getFP(),dstClassifier.getFN(),
                            dstClassifier.getAccuracy(),dstClassifier.getAUC()));
            data.add(Arrays.asList("RF",rfClassifier.getTotalTrue(),rfClassifier.getTotalFalse(),
                            rfClassifier.getTrainTime(),rfClassifier.getPredictTime(),
                            rfClassifier.getTP(),rfClassifier.getTN(),rfClassifier.getFP(),rfClassifier.getFN(),
                            rfClassifier.getAccuracy(),rfClassifier.getAUC()));
            data.add(Arrays.asList("NB",nbClassifier.getTotalTrue(),nbClassifier.getTotalFalse(),
                            nbClassifier.getTrainTime(),nbClassifier.getPredictTime(),
                            nbClassifier.getTP(),nbClassifier.getTN(),nbClassifier.getFP(),nbClassifier.getFN(),
                            nbClassifier.getAccuracy(),nbClassifier.getAUC()));
            data.add(Arrays.asList("LR",lrClassifier.getTotalTrue(),lrClassifier.getTotalFalse(),
                            lrClassifier.getTrainTime(),lrClassifier.getPredictTime(),
                            lrClassifier.getTP(),lrClassifier.getTN(),lrClassifier.getFP(),lrClassifier.getFN(),
                            lrClassifier.getAccuracy(),lrClassifier.getAUC()));

            System.out.println("-----------------------------------------------------------------------------");
            System.out.println("Classifiers Analysis results");
            System.out.println("-----------------------------------------------------------------------------");
            System.out.printf(columnNames.toString());
            System.out.println();
            for(Object object: data){
                System.out.format(object.toString());
                System.out.println();
            }
            System.out.println("-----------------------------------------------------------------------------");
            System.out.println("Total execution time: " + (System.currentTimeMillis()-start)/1000 +"seconds");
            System.out.println("-----------------------------------------------------------------------------");
        
    }
}
