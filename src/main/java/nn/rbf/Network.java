package nn.rbf;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Network {

  private static final double BETA = 2.0;
  private static final double MAX_DISTANCE_PROPORTION = 1.0;

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

  public Network(final int neuronsin,
                 final int neuronshidden,
                 final int neuronsout,
                 final double[][] trainin, final double[][] trainout, final double[][] testin, final double[][] testout) {
    this.neuronsin = neuronsin;
    this.neuronshidden = neuronshidden;
    this.neuronsout = neuronsout;
    this.trainin = trainin;
    this.trainout = trainout;
    this.testin = testin;
    this.testout = testout;

    if (neuronshidden > trainin.length) {
      throw new RuntimeException("Too many neurons in hidden layer");
    }

    init();
  }

  public void learn(final int maxIterations, double learningRate) {

    initHiddenLayer();

    double variance = calcMaxDistance(hidden) * MAX_DISTANCE_PROPORTION;
    double beta = 1.0/(2.0*variance);

    for (int i=0, count=0; i < maxIterations; i++) {
      double error = 0.0;
      for (int j=0; j < trainin.length; j++) {
        double[] invalues = trainin[j];
        for (int k=0; k < in.size(); k++) {
          in.get(k).setOutput(invalues[k]);
        }

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
          out.get(k).applyBackpropagation(trainout[j][k], learningRate);
        }
      }

      if (count % 100 == 0) {
        System.out.printf("EPOCH: %d, error: %f\n", i, error);
      }
      count++;
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
}
