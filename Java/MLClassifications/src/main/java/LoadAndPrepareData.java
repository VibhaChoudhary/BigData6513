import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import javax.swing.*;

public class LoadAndPrepareData {
    //load and prepare dataset
    public Dataset<Row>[]  load(SparkSession spark, String filePath){

        StructField[] structFields = new StructField[]{
                new StructField("Age", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("Attrition", DataTypes.StringType,true,Metadata.empty()),
                new StructField("BusinessTravel", DataTypes.StringType,true,Metadata.empty()),
                new StructField("DailyRate", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("Department",DataTypes.StringType,true,Metadata.empty()),
                new StructField("DistanceFromHome", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("Education", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("EducationField", DataTypes.StringType,true,Metadata.empty()),
                new StructField("EmployeeCount", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("EmployeeNumber", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("EnvironmentSatisfaction", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("Gender", DataTypes.StringType,true,Metadata.empty()),
                new StructField("HourlyRate", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("JobInvolvement", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("JobLevel", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("JobRole", DataTypes.StringType,true,Metadata.empty()),
                new StructField("JobSatisfaction", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("MaritalStatus", DataTypes.StringType,true,Metadata.empty()),
                new StructField("MonthlyIncome", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("MonthlyRate", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("NumCompaniesWorked", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("Over18", DataTypes.StringType,true,Metadata.empty()),
                new StructField("OverTime", DataTypes.StringType,true,Metadata.empty()),
                new StructField("PercentSalaryHike", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("PerformanceRating", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("RelationshipSatisfaction", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("StandardHours",DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("StockOptionLevel", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("TotalWorkingYears", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("TrainingTimesLastYear", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("WorkLifeBalance", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("YearsAtCompany", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("YearsInCurrentRole", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("YearsSinceLastPromotion", DataTypes.IntegerType,true,Metadata.empty()),
                new StructField("YearsWithCurrManager", DataTypes.IntegerType,true,Metadata.empty())
        };

        StructType structType = new StructType(structFields);
        //String filePath =  LoadAndPrepareData.class.getResource("attrition_data.csv").getPath().replace("JavaDecisionTree-1.0-SNAPSHOT.jar!/","");
        Dataset<Row> data = spark.read().format("csv").option("header","true")
                .option("delimiter", ",")
                .schema(structType)
                .load(filePath);
        data.na().drop();
        StringIndexer Attrition_LabelIndexer = new StringIndexer().
                setInputCol("Attrition").setOutputCol("label");
        StringIndexer BusinessTravel_LabelIndexer = new StringIndexer().
                setInputCol("BusinessTravel").setOutputCol("BusinessTravelI");
        StringIndexer Department_LabelIndexer = new StringIndexer().
                setInputCol("Department").setOutputCol("DepartmentI");
        StringIndexer EducationField_LabelIndexer = new StringIndexer().
                setInputCol("EducationField").setOutputCol("EducationFieldI");
        StringIndexer Gender_LabelIndexer = new StringIndexer().
                setInputCol("Gender").setOutputCol("GenderI");
        StringIndexer JobRole_LabelIndexer = new StringIndexer().
                setInputCol("JobRole").setOutputCol("JobRoleI");
        StringIndexer MaritalStatus_LabelIndexer = new StringIndexer().
                setInputCol("MaritalStatus").setOutputCol("MaritalStatusI");
        StringIndexer Over18_LabelIndexer = new StringIndexer().
                setInputCol("Over18").setOutputCol("Over18I");
        StringIndexer OverTime_LabelIndexer = new StringIndexer().
                setInputCol("OverTime").setOutputCol("OverTimeI");

        VectorAssembler assembler = new VectorAssembler().
                setInputCols(new String[]{
                        "BusinessTravelI", "DepartmentI", "EducationFieldI","GenderI",
                        "JobRoleI", "MaritalStatusI", "Over18I", "OverTimeI",
                        "Age","DailyRate", "DistanceFromHome", "Education","EmployeeCount", "EmployeeNumber",
                        "EnvironmentSatisfaction","HourlyRate","JobInvolvement","JobLevel","JobSatisfaction",
                        "MonthlyIncome","MonthlyRate","NumCompaniesWorked","PercentSalaryHike",
                        "PerformanceRating","RelationshipSatisfaction","StandardHours","StockOptionLevel",
                        "TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany",
                        "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"}).
                setOutputCol("features");

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{Attrition_LabelIndexer,BusinessTravel_LabelIndexer,Department_LabelIndexer,
                        EducationField_LabelIndexer,Gender_LabelIndexer,JobRole_LabelIndexer,
                        MaritalStatus_LabelIndexer,Over18_LabelIndexer,OverTime_LabelIndexer,assembler}
        );

        data = pipeline.fit(data).transform(data);

        Dataset<Row>[] splits = data.randomSplit(new double[] {0.70, 0.30},43);
        return splits;
    }

}
