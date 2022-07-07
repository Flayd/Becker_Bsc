import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;

import org.deeplearning4j.datasets.iterator.callbacks.FileCallback;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class NormalizedTraceCallback implements FileCallback{

	public static Boolean successful;
	public static ErrorType errorType;
	
	public <T> T call(File file) {
		
		int scalarPerRow = 11;
		int maxRows = 750;
		int totalArraySize = scalarPerRow * maxRows;
		
		
		try {
		BufferedReader buffer = new BufferedReader(new FileReader(file));
		String line =  buffer.readLine();
		
		if(line.contains("true")) {
			successful =true;
		}else {
			successful=false;
		}
		
		if(line.contains("NONE")) {
			
			errorType = ErrorType.NONE;
			
		}else if(line.contains(ErrorType.SELFGENERATED.toString())){
				
				errorType = ErrorType.SELFGENERATED;
			

		}else if(line.contains("NULL")){
					
				if(line.contains(ErrorType.NULLACCESS.toString())) {
					
					errorType = ErrorType.NULLACCESS;
					
				}else if(line.contains(ErrorType.NULLLENGTH.toString())) {
					
					errorType = ErrorType.NULLLENGTH;
					
				}else if(line.contains(ErrorType.NULLARRAY.toString())) {
					
					errorType = ErrorType.NULLARRAY;
					
				}else if(line.contains(ErrorType.NULLTHROW.toString())){
					
					errorType = ErrorType.NULLTHROW;
					
				}
			
		}else if(line.contains("IndexOutOfBounds")) {
			
			if(line.contains(ErrorType.IndexOutOfBoundsArrayList.toString())) {
				
				errorType = ErrorType.IndexOutOfBoundsArrayList;
				
			}else if(line.contains(ErrorType.IndexOutOfBoundsString.toString())) {
				
				errorType = ErrorType.IndexOutOfBoundsString;
			}
			
		}else {
			errorType = ErrorType.RANDOM;
		}

		
		line = buffer.readLine();
		
		ArrayList<Double> scalarList = new ArrayList<Double>();
		int rescaleFactor = 1000;
		while(line != null) {
		
			
			String[] lineSplit = line.split("\t");
			for(int i=0;i<lineSplit.length;i++) {
				scalarList.add((Double.parseDouble(lineSplit[i])/rescaleFactor));
			}
			
			int currentRowlength = scalarList.size() % scalarPerRow;
			
			if(currentRowlength !=0) {
				//Normalizing length of each row by adding "-0.001" (length = scalarPerRow)
				for(int i =currentRowlength+1;i<=scalarPerRow;i++) {
					scalarList.add(-1.0/rescaleFactor);
				}
			}
			line = buffer.readLine();
		}
		buffer.close();
			
			//Normalizing the amount of rows by Adding "-0.001" (amount = maxRows)
			
			int listLength = scalarList.size();
			for(int i = listLength;i<=totalArraySize;i++) {
				scalarList.add(-1.0/rescaleFactor);
			}
			
			INDArray input = Nd4j.create(scalarList.size()/scalarPerRow,scalarPerRow);
//			INDArray input = Nd4j.create(1,scalarList.size()-1);
//			int[] shape = {scalarList.size()/scalarPerRow,scalarPerRow};
//			input = input.reshape(shape);
			
			
			
			for(int i=0;i<scalarList.size()-1;i++) {
				input.putScalar(i,scalarList.get(i));
			
			}

			DataSet ds = new DataSet(input,input);

			return  (T) ds;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return null;
			
		} catch (IOException e) {
			
			e.printStackTrace();
			return null;
		}
		
		
	}



}
