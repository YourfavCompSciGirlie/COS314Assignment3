import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.attributeSelection.InfoGainAttributeEval;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class BTCDecisionTreeEngineered {

    public static void main(String[] args) {
        System.setProperty("weka.core.WekaPackageManager.disablePackageManager", "true");

        try {
            System.out.println("=== BTC Decision Tree with Feature Engineering ===");

            Scanner scanner = new Scanner(System.in);
            System.out.print("Enter training file path: ");
            String trainPath = scanner.nextLine();
            System.out.print("Enter test file path: ");
            String testPath = scanner.nextLine();
            System.out.print("Enter seed value: ");
            int seed = scanner.nextInt();
            scanner.close();

            PrintWriter results = new PrintWriter(new FileWriter("engineered_results.txt"));
            DecimalFormat df = new DecimalFormat("#.####");

            // Loading datasets
            DataSource trainSource = new DataSource(new File(trainPath).getAbsolutePath());
            Instances originalTrainData = trainSource.getDataSet();
            DataSource testSource = new DataSource(new File(testPath).getAbsolutePath());
            Instances originalTestData = testSource.getDataSet();

            originalTrainData.setClassIndex(originalTrainData.numAttributes() - 1);
            originalTestData.setClassIndex(originalTestData.numAttributes() - 1);

            System.out.println("\n=== ORIGINAL DATA ANALYSIS ===");
            System.out.println("Original attributes: " + originalTrainData.numAttributes());
            for (int i = 0; i < originalTrainData.numAttributes(); i++) {
                System.out.println("  " + originalTrainData.attribute(i).name());
            }

            // Creating feature engineering 
            System.out.println("\n=== FEATURE ENGINEERING ===");
            Instances trainData = createEngineeredFeatures(originalTrainData);
            Instances testData = createEngineeredFeatures(originalTestData);
            
            System.out.println("Engineered attributes: " + trainData.numAttributes());
            results.println("=== ENGINEERED FEATURES ===");
            for (int i = 0; i < trainData.numAttributes(); i++) {
                String attrName = trainData.attribute(i).name();
                System.out.println("  " + attrName);
                results.println("  " + attrName);
            }

            // Convert class to nominal
            if (trainData.classAttribute().isNumeric()) {
                NumericToNominal convert = new NumericToNominal();
                convert.setAttributeIndices("" + (trainData.classIndex() + 1));
                convert.setInputFormat(trainData);
                trainData = Filter.useFilter(trainData, convert);
                testData = Filter.useFilter(testData, convert);
            }

            // Handle missing values
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(trainData);
            trainData = Filter.useFilter(trainData, replaceMissing);
            testData = Filter.useFilter(testData, replaceMissing);

            // Information gain analysis
            System.out.println("\n=== INFORMATION GAIN ANALYSIS ===");
            results.println("\n=== INFORMATION GAIN ANALYSIS ===");
            
            InfoGainAttributeEval infoGain = new InfoGainAttributeEval();
            infoGain.buildEvaluator(trainData);
            
            double maxGain = 0;
            String bestAttribute = "";
            for (int i = 0; i < trainData.numAttributes() - 1; i++) {
                double gain = infoGain.evaluateAttribute(i);
                String gainInfo = trainData.attribute(i).name() + ": " + df.format(gain);
                System.out.println("  " + gainInfo);
                results.println("  " + gainInfo);
                
                if (gain > maxGain) {
                    maxGain = gain;
                    bestAttribute = trainData.attribute(i).name();
                }
            }
            
            System.out.println("Best feature: " + bestAttribute + " (gain: " + df.format(maxGain) + ")");
            results.println("Best feature: " + bestAttribute + " (gain: " + df.format(maxGain) + ")");

            // Class distribution
            System.out.println("\n=== CLASS DISTRIBUTION ===");
            results.println("\n=== CLASS DISTRIBUTION ===");
            int[] classCounts = trainData.attributeStats(trainData.classIndex()).nominalCounts;
            for (int i = 0; i < classCounts.length; i++) {
                String classInfo = "Class " + trainData.classAttribute().value(i) + ": " + classCounts[i] + " instances";
                System.out.println(classInfo);
                results.println(classInfo);
            }

            // Train decision tree
            System.out.println("\n=== TRAINING DECISION TREE ===");
            J48 tree = new J48();
            tree.setMinNumObj(10);  // Conservative to prevent overfitting
            tree.setUnpruned(false); // Enable pruning
            tree.setConfidenceFactor(0.25f);
            tree.buildClassifier(trainData);

            System.out.println("\n=== DECISION TREE STRUCTURE ===");
            System.out.println(tree.toString());
            results.println("\n=== DECISION TREE STRUCTURE ===");
            results.println(tree.toString());

            // Cross-validation
            System.out.println("\n=== CROSS-VALIDATION RESULTS ===");
            Evaluation crossVal = new Evaluation(trainData);
            crossVal.crossValidateModel(tree, trainData, 10, new Random(seed));
            
            String cvSummary = crossVal.toSummaryString("=== 10-Fold Cross-Validation ===\n", false);
            String cvDetails = crossVal.toClassDetailsString();
            String cvMatrix = crossVal.toMatrixString();
            
            System.out.println(cvSummary);
            System.out.println(cvDetails);
            System.out.println(cvMatrix);
            
            results.println(cvSummary);
            results.println(cvDetails);
            results.println(cvMatrix);

            // Test set evaluation
            System.out.println("\n=== TEST SET RESULTS ===");
            Evaluation testEval = new Evaluation(trainData);
            testEval.evaluateModel(tree, testData);
            
            String testSummary = testEval.toSummaryString("=== Test Set Evaluation ===\n", false);
            String testDetails = testEval.toClassDetailsString();
            String testMatrix = testEval.toMatrixString();
            
            System.out.println(testSummary);
            System.out.println(testDetails);
            System.out.println(testMatrix);
            
            results.println(testSummary);
            results.println(testDetails);
            results.println(testMatrix);

            // Performance summary
            System.out.println("\n=== PERFORMANCE SUMMARY ===");
            results.println("\n=== PERFORMANCE SUMMARY ===");
            
            double cvAccuracy = crossVal.pctCorrect();
            double testAccuracy = testEval.pctCorrect();
            double cvPrecision = crossVal.weightedPrecision();
            double testPrecision = testEval.weightedPrecision();
            double cvRecall = crossVal.weightedRecall();
            double testRecall = testEval.weightedRecall();
            double cvF1 = crossVal.weightedFMeasure();
            double testF1 = testEval.weightedFMeasure();
            
            String[] performanceLines = {
                "Cross-Validation Accuracy: " + df.format(cvAccuracy) + "%",
                "Test Set Accuracy: " + df.format(testAccuracy) + "%",
                "Cross-Validation Precision: " + df.format(cvPrecision),
                "Test Set Precision: " + df.format(testPrecision),
                "Cross-Validation Recall: " + df.format(cvRecall),
                "Test Set Recall: " + df.format(testRecall),
                "Cross-Validation F1-Score: " + df.format(cvF1),
                "Test Set F1-Score: " + df.format(testF1),
                "Tree Leaves: " + tree.measureNumLeaves(),
                "Tree Size: " + tree.measureTreeSize(),
                "Best Feature: " + bestAttribute,
                "Max Information Gain: " + df.format(maxGain)
            };
            
            for (String line : performanceLines) {
                System.out.println(line);
                results.println(line);
            }

            results.close();
            System.out.println("\nResults saved to: engineered_results.txt");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Create engineered features from OHLC data
     */
    private static Instances createEngineeredFeatures(Instances originalData) {
        try {
           
            ArrayList<Attribute> attributes = new ArrayList<>();
            
            // Technical indicators
            attributes.add(new Attribute("Daily_Return"));          
            attributes.add(new Attribute("High_Low_Range"));         
            attributes.add(new Attribute("Body_Size"));              
            attributes.add(new Attribute("Upper_Shadow"));           
            attributes.add(new Attribute("Lower_Shadow"));           
            attributes.add(new Attribute("Price_Position"));         
            attributes.add(new Attribute("Volatility"));           
            attributes.add(new Attribute("Gap"));                    
            attributes.add(new Attribute("Trend_Strength"));       
            attributes.add(new Attribute("Relative_Volume"));       
            
           
            attributes.add(new Attribute("SMA3_Close"));
            attributes.add(new Attribute("Close_vs_SMA3"));
            
            // Price ratios
            attributes.add(new Attribute("Close_to_High_Ratio"));   
            attributes.add(new Attribute("Close_to_Low_Ratio"));     
            attributes.add(new Attribute("Open_to_Close_Ratio"));    
            
            
            ArrayList<String> classValues = new ArrayList<>();
            for (int i = 0; i < originalData.classAttribute().numValues(); i++) {
                classValues.add(originalData.classAttribute().value(i));
            }
            if (originalData.classAttribute().isNumeric()) {
                attributes.add(new Attribute("Output"));
            } else {
                attributes.add(new Attribute("Output", classValues));
            }

            
            Instances engineeredData = new Instances("EngineeredBTC", attributes, originalData.numInstances());
            engineeredData.setClassIndex(engineeredData.numAttributes() - 1);

            // Calculating engineered features for each instance
            for (int i = 0; i < originalData.numInstances(); i++) {
                Instance original = originalData.instance(i);
                
                double open = original.value(0);    
                double high = original.value(1);    
                double low = original.value(2);    
                double close = original.value(3);   
                double adjClose = original.value(4); 
                double output = original.value(5);   
                
              
                double[] values = new double[engineeredData.numAttributes()];
                int idx = 0;
                
                
                values[idx++] = (close - open) / Math.max(open, 0.0001);  
                values[idx++] = (high - low) / Math.max(open, 0.0001);   
                values[idx++] = Math.abs(close - open) / Math.max(high - low, 0.0001); 
                values[idx++] = (high - Math.max(open, close)) / Math.max(high - low, 0.0001); 
                values[idx++] = (Math.min(open, close) - low) / Math.max(high - low, 0.0001); 
                values[idx++] = (close - low) / Math.max(high - low, 0.0001); 
                values[idx++] = (high - low) / Math.max(close, 0.0001);   
                
                
                if (i > 0) {
                    double prevClose = originalData.instance(i-1).value(3);
                    values[idx++] = (open - prevClose) / Math.max(prevClose, 0.0001);
                } else {
                    values[idx++] = 0.0;
                }
                
                values[idx++] = Math.abs(close - open) / Math.max(close, 0.0001); 
                values[idx++] = (high - low) / Math.max(adjClose, 0.0001);      
                
                // Simple moving averages (3-period)
                double sma3 = close;
                if (i >= 2) {
                    sma3 = (close + originalData.instance(i-1).value(3) + originalData.instance(i-2).value(3)) / 3.0;
                }
                values[idx++] = sma3;
                values[idx++] = (close - sma3) / Math.max(sma3, 0.0001);
                
                
                values[idx++] = close / Math.max(high, 0.0001);
                values[idx++] = close / Math.max(low, 0.0001);
                values[idx++] = open / Math.max(close, 0.0001);
                
               
                values[idx++] = output;
                
              
                Instance newInstance = new DenseInstance(1.0, values);
                engineeredData.add(newInstance);
            }
            
            return engineeredData;
            
        } catch (Exception e) {
            System.err.println("Feature engineering failed: " + e.getMessage());
            return originalData;
        }
    }
}