package nn.rbf;

import nn.rbf.utils.DataSetUtils;
import nn.rbf.utils.FileUtils;

public class Main {
  public static void main(String[] args) {
    // APPROXIMATION
    double[][][] trainDS = FileUtils.loadData("data/approximation_train_3.txt", FileUtils.SPACE, 1, 1, false);
    // CLASSIFICATION 1
    //double[][][] trainDS = FileUtils.loadData("data/classification_train.txt", FileUtils.SPACE, 3, 1, false);
    // CLASSIFICATION 2
    //double[][][] trainDS = FileUtils.loadData("data/classification_train.txt", FileUtils.SPACE, 3, 3, true);
    if (trainDS == null) {
      throw new RuntimeException("Wrong input dataset");
    }
    double[][] trainIn = trainDS[0];
    double[][] trainOut = trainDS[1];
    DataSetUtils.normalizeInRangeZeroOne(trainIn);
    DataSetUtils.normalizeInRangeZeroOne(trainOut);

    // APPROXIMATION
    double[][][] testDS = FileUtils.loadData("data/approximation_test.txt", FileUtils.SPACE, 1, 1, false);
    // CLASSIFICATION 1
    //double[][][] testDS = FileUtils.loadData("data/classification_test.txt", FileUtils.SPACE, 3, 1, false);
    // CLASSIFICATION 2
    //double[][][] testDS = FileUtils.loadData("data/classification_test.txt", FileUtils.SPACE, 3, 3, true);
    if (testDS == null) {
      throw new RuntimeException("Wrong input dataset");
    }
    double[][] testIn = null;
    double[][] testOut = null;
    if (testDS != null) {
      testIn = testDS[0];
      testOut = testDS[1];
      DataSetUtils.normalizeInRangeZeroOne(testIn);
      DataSetUtils.normalizeInRangeZeroOne(testOut);
    }

    // Approximation
    Network network = new Network(1, 4, 1, trainIn, trainOut, testIn, testOut, true, false);
    network.learn(10000, 0.05);

    // Classification 1
    //Network network = new Network(3, 24, 1, trainIn, trainOut, testIn, testOut, false, true);
    //network.learn(600000, 0.04);

    // Classification 2
    //Network network = new Network(3, 24, 3, trainIn, trainOut, testIn, testOut, false, true);
    //network.learn(100000, 0.009);
  }
}
