package nn.rbf.utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FileUtils {
  public static final String SPACE = " ";
  public static final String COMMA = ",";

  public static double[][] loadData(String fileName, String separator) {
    File file = new File(fileName);
    String[] tempTable;
    double[] tempList;
    int rows = 0;
    try{
      FileReader fr = new FileReader(file);
      BufferedReader input = new BufferedReader(fr);
      String line;
      System.out.println("Loading data from: \"" + fileName + "\"...");
      List<double[]> dataList = new ArrayList<>();
      while((line = input.readLine()) != null) {
        rows ++;
        tempTable = line.split(separator);
        int tableLenght = tempTable.length;
        tempList = new double[tableLenght];
        for(int i = 0; i< tableLenght; i++){
          tempList[i] = Double.valueOf(tempTable[i]);
        }
        dataList.add(tempList);
      }
      fr.close();
      System.out.println(rows + " rows was imported");
      return dataList.toArray(new double[][]{});
    }catch(Exception e){
      System.out.println("File can not be read!. Error: " + e);
    }
    return null;
  }

  public static double[][][] loadData(String fileName, String separator, int numInputs, int numOutputs, boolean convertOutputTo01) {
    File file = new File(fileName);
    int rows = 0;
    try{
      FileReader fr = new FileReader(file);
      BufferedReader input = new BufferedReader(fr);
      String line;
      System.out.println("Loading data from: \"" + fileName + "\"...");
      List<double[]> inData = new ArrayList<>();
      List<double[]> outData = new ArrayList<>();
      while((line = input.readLine()) != null) {
        rows ++;
        String[] tempTable = line.split(separator);
        int tableLenght = tempTable.length;
        double[] in = new double[numInputs];
        double[] out = new double[numOutputs];
        for(int i = 0; i < numInputs; i++){
          in[i] = Double.valueOf(tempTable[i]);
        }
        if (convertOutputTo01) {
          int outv = Integer.valueOf(tempTable[tableLenght-1]);
          if (outv < 0 || outv > numOutputs) {
            throw new IllegalArgumentException("Invalid output feature value. Not inline with number of output neurons");
          }
          Arrays.fill(out, 0.0);
          out[outv-1] = 1.0;
        } else {
          for (int i = tableLenght - 1, j = 0; i > (tableLenght - 1 - numOutputs); i--) {
            out[j++] = Double.valueOf(tempTable[i]);
          }
        }
        inData.add(in);
        outData.add(out);
      }
      fr.close();
      System.out.println(rows + " rows was imported");
      return new double[][][]{inData.toArray(new double[][]{}), outData.toArray(new double[][]{})};
    }catch(Exception e){
      System.out.println("File can not be read!. Error: " + e);
    }
    return null;
  }

//  public static void saveNetworkToFile(Network network, String fileName){
//    File outFile =  new File(fileName);
//    try{
//      FileWriter fw = new FileWriter(outFile);
//      PrintWriter pw = new PrintWriter(fw);
//      int networkSize = network.getNumOfNeurons();
//      for (int i = 0; i < networkSize; i++) {
//        String weightsLine = "";
//        double[] weights = network.getNeuron(i).getWeights();
//        for (int j=0; j < weights.length; j++) {
//          weightsLine += weights[j];
//          if (j < weights.length - 1){
//            weightsLine += ",";
//          }
//        }
//        pw.println(weightsLine);
//      }
//      fw.close();
//    }catch(Exception e){
//      System.out.println("File can not be read!. Error: " + e);
//    }
//  }

}
