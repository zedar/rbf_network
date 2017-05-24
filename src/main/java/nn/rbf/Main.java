package nn.rbf;

import nn.rbf.utils.DataSetUtils;
import nn.rbf.utils.FileUtils;

public class Main {
  public static void main(String[] args) {
    double[][][] trainDS = FileUtils.loadData("data/approximation_train_3.txt", FileUtils.SPACE, 1, 1);
    if (trainDS == null) {
      throw new RuntimeException("Wrong input dataset");
    }
    double[][] trainIn = trainDS[0];
    double[][] trainOut = trainDS[1];
    DataSetUtils.normalizeInRangeZeroOne(trainIn);
    DataSetUtils.normalizeInRangeZeroOne(trainOut);

    double[][][] testDS = FileUtils.loadData("data/approximation_test.txt", FileUtils.SPACE, 1, 1);
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

    Network network = new Network(1, 24, 1, trainIn, trainOut, testIn, testOut, true);

    network.learn(100000, 0.05);
  }
}
