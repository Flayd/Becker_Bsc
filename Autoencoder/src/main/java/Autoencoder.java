import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;


import org.deeplearning4j.datasets.iterator.FileSplitDataSetIterator;
import org.deeplearning4j.datasets.iterator.callbacks.FileCallback;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Autoencoder {
    
	public static Map<DataSet,Boolean> expectedSuccess = new Hashtable<DataSet,Boolean>();
	public static Map<DataSet,ErrorType> errorTypeMap = new Hashtable<DataSet,ErrorType>();
	

	
	public static void main(String[] args) throws IOException {
		
		//takes all Files in the given folder and passes them to the Autoencoder
		File traingFolder = new File("/home/ubuntu/eclipse-workspace/ExTra/extra-base/ExTra/TrainingTraces/");
		File testingFolder = new File("/home/ubuntu/eclipse-workspace/ExTra/extra-base/ExTra/TestTraces/");
		List<File>  trainingTracesInFolder = Arrays.asList(traingFolder.listFiles());
		List<File>  testingTracesInFolder = Arrays.asList(testingFolder.listFiles());
		
		

		FileCallback callback = new NormalizedTraceCallback();
			
		DataSetIterator fileTrain = new FileSplitDataSetIterator(trainingTracesInFolder, callback);
		DataSetIterator fileTest = new FileSplitDataSetIterator(testingTracesInFolder, callback);
		
		final int inputChannels = 11;
		int outputNum = 11;
//		int batchSize =  128;
		int rngSeed = 123;
		int numEpochs = 3; 
		double learningRate = 0.05; //normal value either 0.01 or 0.05


		@SuppressWarnings("deprecation")
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(rngSeed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Sgd(learningRate))
				.l2(1e-4).list()
				

				.layer(0,new DenseLayer.Builder()
						.nIn(inputChannels).nOut(8)		
						.activation(Activation.LEAKYRELU)
						.weightInit(WeightInit.RELU)
						.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) 
						.build())
				
				.layer(1,new DenseLayer.Builder()
						.nIn(8).nOut(6)		
						.activation(Activation.LEAKYRELU)
						.weightInit(WeightInit.RELU)
						.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) 
						.build())
				
				.layer(2,new DenseLayer.Builder()
						.nIn(6).nOut(8)	
						.activation(Activation.LEAKYRELU)
						.weightInit(WeightInit.RELU)
						.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) 
						.build())
				
				.layer(3,new OutputLayer.Builder(LossFunction.L2)
						.nIn(8).nOut(outputNum)
						.activation(Activation.LEAKYRELU)
						.weightInit(WeightInit.RELU)
						.build())
				
				.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		System.out.println(model.summary());
		

		model.setListeners(new ScoreIterationListener(50));
		
		//training the Autoencoder with the list of fileTrain
		for(int i=0;i<numEpochs;i++) {
			model.fit(fileTrain); //fit kann auch numEpochs als Eingabe nehmen	
		
		}
		

			fileTrain.reset();
			double max = 0;
			double addedValues = 0;
			int trainingFileAmount = trainingTracesInFolder.size();
			while(fileTrain.hasNext()) {
				
			DataSet next = fileTrain.next();
			
			double currentValue = model.score(next);
			addedValues = addedValues + currentValue;
			
			if(currentValue>max) {
				max = currentValue;
			}
			
		
		}
			double avgErrorScore = addedValues / trainingFileAmount;
			
			System.out.println("\nMax Score for trainingData: " + max);
			System.out.println("\nAvg Score for trainingData: " + avgErrorScore);
			
			
			int nullAccessCount=0;
			int nullLengthCount=0;
			int nullArrayCount=0;
			int nullThrowCount=0;
			int selfgeneratedCount=0;
			int indexOutOfBoundsStringCount=0;
			int indexOutOfBoundsArrayListCount=0;
			
			while(fileTest.hasNext()) {

				DataSet next = fileTest.next();
				
				
				expectedSuccess.put(next, NormalizedTraceCallback.successful);
				errorTypeMap.put(next, NormalizedTraceCallback.errorType);
				
				ErrorType error = NormalizedTraceCallback.errorType;
				
				if(error!= ErrorType.NONE) {
					if(error == ErrorType.NULLACCESS) {
						nullAccessCount++;
					}else if(error == ErrorType.NULLLENGTH) {
						nullLengthCount++;
					}else if(error == ErrorType.NULLARRAY) {
						nullArrayCount++;
					}else if(error == ErrorType.NULLTHROW) {
						nullThrowCount++;
					}else if(error == ErrorType.IndexOutOfBoundsArrayList) {
						indexOutOfBoundsArrayListCount++;
					}else if(error == ErrorType.IndexOutOfBoundsString) {
						indexOutOfBoundsStringCount++;
					}else if(error == ErrorType.SELFGENERATED) {
						selfgeneratedCount++;
					}
				}
				
			}
			
			boolean dynemic = false;
			double bestTolerance=0;
			
//		if(dynemic) {	
//			double tolerance = 0;
//			double bestAccuracy =0;
//			double bestPrecision =0;
//			double bestRecall =0;
//			
//			
//			while(tolerance <= 1) {
//				double truePositive = 0;
//				double falsePositive = 0;
//				double trueNegative = 0;
//				double falseNegative = 0;
//				
//				for(Map.Entry<DataSet, Boolean> entry: expectedSuccess.entrySet()) {
//					double score = model.score(entry.getKey());
//					double toleranceZone = max*tolerance;
//					boolean predicted = score< (max + toleranceZone);
//			
//					if(predicted==entry.getValue()) {
//						
//						if(entry.getValue()) {
//							truePositive++;
//						}else {
//							trueNegative++;
//						}				
//					}else {
//						
//						if(entry.getValue()==true) {
//							falseNegative++;
//						}else {
//							falsePositive++;
//						}
//					}			
//				}
//				
//				double precision = (truePositive/(truePositive + falsePositive));
//				double recall = (truePositive/(truePositive + falseNegative));
//				double simpleAccuracy = ((truePositive + trueNegative)/(truePositive +trueNegative + falsePositive + falseNegative)); 
//				
//				if((simpleAccuracy > bestAccuracy) && (precision >= bestPrecision || recall >= bestRecall)) {
//					
//					bestAccuracy = simpleAccuracy;
//					bestPrecision = precision;
//					bestRecall = recall;
//					System.out.println("new best Precision: " + bestPrecision + " Recall: " + bestRecall + " Accuracy: " + bestAccuracy +" at tolerance " + tolerance);
//					bestTolerance = tolerance;
//					
//				}
//				tolerance = tolerance + 0.01;
//				
//			}
//		}else {
			bestTolerance = 0.25;
//		}
		
			
			double truePositive = 0;
			double falsePositive = 0;
			double trueNegative = 0;
			double falseNegative = 0;
			double maxScorePositive=0;
			double minScoreNegative=1;
			
			int nullAccessCountCorrect=0;
			int nullLengthCountCorrect=0;
			int nullArrayCountCorrect=0;
			int nullThrowCountCorrect=0;
			int indexOutOfBoundsStringCorrect=0;
			int indexOutOfBoundsArrayListCorrect=0;
			int selfgeneratedCorrect=0;
			
		
			for(Map.Entry<DataSet, Boolean> entry: expectedSuccess.entrySet()) {
				double score = model.score(entry.getKey());
				double toleranceZone = max*bestTolerance; // possible formular: max*bestTolerance ; avgErrorScore*bestTolerance
				boolean predicted = score < (max + toleranceZone); //For false training date score > rest
				
				//Calculate highest true Trace score and lowest wrong Trace score for reference.
				if(entry.getValue()) {
					
					if(maxScorePositive<score) {
						
						maxScorePositive = score;
					}
				}else {
					
					if(score<minScoreNegative) {
						minScoreNegative = score;
					}
				}
				System.out.println("Testing Score: " + score);
							
				if(predicted==entry.getValue()) {
					System.out.println("Success was predicted correctly: predicted: " + predicted + "; expected: " + entry.getValue());
					
					if(errorTypeMap.get(entry.getKey())== ErrorType.NULLACCESS) {
						nullAccessCountCorrect++;
					}else if(errorTypeMap.get(entry.getKey())== ErrorType.NULLLENGTH) {
						nullLengthCountCorrect++;
					}else if(errorTypeMap.get(entry.getKey())== ErrorType.NULLARRAY) {
						nullArrayCountCorrect++;
					}else if(errorTypeMap.get(entry.getKey())== ErrorType.NULLTHROW) {
						nullThrowCountCorrect++;
					}else if(errorTypeMap.get(entry.getKey())== ErrorType.IndexOutOfBoundsArrayList) {
						indexOutOfBoundsArrayListCorrect++;
					}else if(errorTypeMap.get(entry.getKey())== ErrorType.IndexOutOfBoundsString) {
						indexOutOfBoundsStringCorrect++;
					}else if(errorTypeMap.get(entry.getKey())== ErrorType.SELFGENERATED){
						selfgeneratedCorrect++;
					}
					
					if(entry.getValue()) {
						truePositive++;
						
						
					}else {
						trueNegative++;
					}				
				}else {
					System.out.println("Success was not predicted correctly: predicted: " + predicted + "; expected: " + entry.getValue());
					
					if(entry.getValue()==true) {
						falseNegative++;
					}else {
						falsePositive++;
					}
				}			
			}
			
			double precision = (truePositive/(truePositive + falsePositive));
			double recall = (truePositive/(truePositive + falseNegative));
			double simpleAccuracy = ((truePositive + trueNegative)/(truePositive +trueNegative + falsePositive + falseNegative)); 
			
			
			System.out.println("-----------------------------------------------------------------------------------------------------------");
			System.out.println("Selfgenerated-Error appeared " + selfgeneratedCount + " times and got predicted correctly " + selfgeneratedCorrect + " times");
			System.out.println("-----------------------------------------------------------------------------------------------------------");
			System.out.println("IndexOutOfBoundsExceptions:");
			System.out.println("IndexOutOfBoundsString-Error appeared " + indexOutOfBoundsStringCount + " times and got predicted correctly " + indexOutOfBoundsStringCorrect + " times");
			System.out.println("IndexOutOfBoundsArrayList-Error appeared " + indexOutOfBoundsArrayListCount + " times and got predicted correctly " + indexOutOfBoundsArrayListCorrect + " times");
			System.out.println("-----------------------------------------------------------------------------------------------------------");
			System.out.println("NullPointerExceptions:");
			System.out.println("NullAccess-Error appeared " + nullAccessCount + " times and got predicted correctly " + nullAccessCountCorrect + " times");
			System.out.println("NullLength-Error appeared " + nullLengthCount + " times and got predicted correctly " + nullLengthCountCorrect + " times");
			System.out.println("NullArray-Error appeared " + nullArrayCount + " times and got predicted correctly " + nullArrayCountCorrect + " times");
			System.out.println("NullThrow-Error appeared " + nullThrowCount + " times and got predicted correctly " + nullThrowCountCorrect + " times");
			System.out.println("-----------------------------------------------------------------------------------------------------------");
			System.out.println("True positive: " + (int)truePositive);
			System.out.println("False positive: " + (int)falsePositive);
			System.out.println("True negative: " + (int)trueNegative);
			System.out.println("False negative: " + (int)falseNegative);
			System.out.println("Precision: " + precision);
			System.out.println("Recall: " + recall);
			System.out.println("Simple Accuracy: " + simpleAccuracy);
			System.out.println("Total NullPointer-Error appeared " + (nullAccessCount+ nullLengthCount + nullArrayCount + nullThrowCount) + " times and got predicted correctly " + (nullAccessCountCorrect + nullLengthCountCorrect + nullArrayCountCorrect + nullThrowCountCorrect) + " times");
			System.out.println("Total IndexOutOfBoundsArrayList-Error appeared " + indexOutOfBoundsArrayListCount + " times and got predicted correctly " + indexOutOfBoundsArrayListCorrect + " times");
			System.out.println("Highest Score for right Traces: " + maxScorePositive);
			System.out.println("Lowest Score for wrong Traces: " + minScoreNegative);
			System.out.println("Range between highest true and lowest false: " + (minScoreNegative - maxScorePositive));
	}
}

