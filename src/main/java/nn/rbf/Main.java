package nn.rbf;

import nn.rbf.utils.DataSetUtils;
import nn.rbf.utils.FileUtils;

public class Main {
  public static void main(String[] args) {
    double[][][] trainDS = FileUtils.loadData("data/approximation_train_1.txt", FileUtils.SPACE, 1, 1);
    if (trainDS == null) {
      throw new RuntimeException("Wrong input dataset");
    }
    double[][] trainIn = trainDS[0];
    double[][] trainOut = trainDS[1];

    DataSetUtils.normalizeInRangeZeroOne(trainIn);
    DataSetUtils.normalizeInRangeZeroOne(trainOut);

    Network network = new Network(1, 10, 1, trainIn, trainOut, null, null);

    network.learn(2000, 0.8);
  }
}
