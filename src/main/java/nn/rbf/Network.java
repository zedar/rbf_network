package nn.rbf;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Network {

  private static final double BETA = 2.0;
  private static final double MAX_DISTANCE_PROPORTION = 1.0;
  private static final double MAX_ERROR = 0.01;

  private final Random rand = new Random();

  private final int neuronsin;
  private final int neuronshidden;
  private final int neuronsout;
  private final double[][] trainin;
  private final double[][] trainout;
  private final double[][] testin;
  private final double[][] testout;
  private final List<BasicNeuron> in = new ArrayList<>();
  private final List<RadialNeuron> hidden = new ArrayList<>();
  private final List<BasicNeuron> out = new ArrayList<>();
  private final boolean printResult;

  public Network(final int neuronsin,
                 final int neuronshidden,
                 final int neuronsout,
                 final double[][] trainin, final double[][] trainout, final double[][] testin, final double[][] testout,
                 boolean printResult) {
    this.neuronsin = neuronsin;
    this.neuronshidden = neuronshidden;
    this.neuronsout = neuronsout;
    this.trainin = trainin;
    this.trainout = trainout;
    this.testin = testin;
    this.testout = testout;
    this.printResult = printResult;

    if (neuronshidden > trainin.length) {
      throw new RuntimeException("Too many neurons in hidden layer");
    }

    init();
  }

  public void learn(final int maxIterations, double learningRate) {

    initHiddenLayer();
    //if (true) return;
    for (int i=0, count=0; i < maxIterations; i++) {
      double error = 0.0;
      for (int j=0; j < trainin.length; j++) {
        double[] invalues = trainin[j];
        for (int k=0; k < in.size(); k++) {
          in.get(k).setOutput(invalues[k]);
        }

        double variance = calcMaxDistance(hidden) * MAX_DISTANCE_PROPORTION;
        double beta = variance; //1.0/(2.0*variance);

        for (RadialNeuron rn : hidden) {
          rn.setBeta(beta);
          rn.calculateOutput();
        }

        for (BasicNeuron bn : out) {
          bn.calculateOutput();
        }

        for (int k=0; k < out.size(); k++) {
          error += Math.pow(out.get(k).getOutput() - trainout[j][k], 2);
        }

        // back propagation
        for (int k=0; k < out.size(); k++) {
          BasicNeuron bn = out.get(k);

//          for (int l=0; l < hidden.size(); l++) {
//            RadialNeuron rn = hidden.get(l);
//            Connection hidden2bn = bn.getConnection(l);
//            rn.applyBackPropagation(bn.getOutput(), trainout[j][k], hidden2bn.getWeight(), learningRate);
//          }

          bn.applyBackpropagation(trainout[j][k], learningRate);
        }
      }

      if (count % 1000 == 0) {
        System.out.printf("EPOCH: %d, error: %f\n", i, error);
      }
      count++;
    }

    if (testin != null && testout != null) {
      test();
    }

  }

  private void init() {
    for (int i=0; i < neuronsin; i++) {
      in.add(new BasicNeuron(BasicNeuron.ACTIVATION_FUNC.LINEAR));
    }

    for (int i=0; i < neuronshidden; i++) {
      RadialNeuron rn = new RadialNeuron(BETA);
      rn.addInConnections(in);
      hidden.add(rn);
    }

    BasicNeuron bias = new BasicNeuron(BasicNeuron.ACTIVATION_FUNC.LINEAR, 1.0);
    for (int i=0; i < neuronsout; i++) {
      BasicNeuron bn = new BasicNeuron(BasicNeuron.ACTIVATION_FUNC.LINEAR);
      bn.addInConnections(hidden);
      bn.addBiasConnection(bias);
      out.add(bn);
    }
  }

  private void initHiddenLayer() {
    for (RadialNeuron rn : hidden) {
      int pos = rand.nextInt(trainin.length);
      rn.setMu(trainin[pos]);
    }
    KMeans.classify2(hidden.size(), trainin, trainout, hidden, 0.001, 50000, true);
  }

  private double calcMaxDistance(List<RadialNeuron> neurons) {
    double maxDistance = 0.0;
    for (int i=0; i < neurons.size(); i++) {
      RadialNeuron n = neurons.get(i);
      for (int j=i+1; j < neurons.size(); j++) {
        double dist = EuclideanDistance.calc(n.getMu(), neurons.get(j).getMu());
        if (dist > maxDistance) {
          maxDistance = dist;
        }
      }
    }
    return maxDistance;
  }

  private double test() {
    PrintWriter fTestExpected = null;
    PrintWriter fTestCalculated = null;
    PrintWriter fAccuracy = null;

    try {
      if (printResult) {
        fTestExpected = new PrintWriter(new FileWriter("out/plot_test_expected.txt"));
        fTestCalculated = new PrintWriter(new FileWriter("out/plot_test_calculated.txt"));
        fAccuracy = new PrintWriter(new FileWriter("out/plot_test_accuracy.txt"));
      }

      int correct = 0;
      for (int i = 0; i < testin.length; i++) {
        double[] invalues = testin[i];
        for (int j = 0; j < in.size(); j++) {
          in.get(j).setOutput(invalues[j]);
        }

        for (RadialNeuron rn : hidden) {
          rn.calculateOutput();
        }

        for (BasicNeuron bn : out) {
          bn.calculateOutput();
        }

        double error = 0.0;
        for (int j = 0; j < out.size(); j++) {
          error += Math.pow(out.get(j).getOutput() - testout[i][j], 2);
        }
        if (error >= -MAX_ERROR && error <= MAX_ERROR) {
          correct++;
        }

        if (printResult) {
          fTestExpected.println("\t" + testin[i][0] + "\t" + testout[i][0]);
          fTestCalculated.println("\t" + testin[i][0] + "\t" + out.get(0).getOutput());
          fAccuracy.println("\t" + i + "\t" + error);
        }
      }
      double accuracy = ((double) correct / (double) testout.length) * 100;
      System.out.printf("CORRECTLY CLASSIFIED: %d ACCURACY: %f \n", correct, accuracy);
      return accuracy;
    } catch (Exception ex) {
      ex.printStackTrace();
      return 0.0;
    } finally {
      if (fTestExpected != null) {
        fTestExpected.close();
      }
      if (fTestCalculated != null) {
        fTestCalculated.close();
      }
      if (fAccuracy != null) {
        fAccuracy.close();
      }
    }
  }
}
