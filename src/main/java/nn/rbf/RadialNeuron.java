package nn.rbf;

public class RadialNeuron extends Neuron {
  private double[] mu;
  private double beta;

  public RadialNeuron(final double beta) {
    this.beta = beta;
  }

  @Override
  public double calculateOutput() {
    double s = 0.0;
    if (in.size() != mu.length) {
      throw new RuntimeException("Wrong number of dimensions");
    }
    for (int i = 0; i < mu.length; i++) {
      double d = in.get(i).getIn().getOutput() - mu[i];
      s += d * d;
    }

    output = Math.exp(-/*beta * */s / (2.0*Math.pow(0.5, 2)));
    return output;
  }

  public void applyBackPropagation(double calculatedOutput, double expectedOutput, double weight, double learningRate) {
    double phi = output;
    double diff = expectedOutput - calculatedOutput;
    for (int i = 0; i < mu.length; i++) {
      mu[i] = mu[i] + learningRate * diff * weight * phi * (in.get(i).getIn().getOutput() - mu[i])/(0.5*0.5);//(beta*beta);
    }
  }

  public double[] getMu() {
    return mu;
  }

  public void setMu(double[] mu) {
    this.mu = mu;
  }

  public double getBeta() {
    return beta;
  }

  public void setBeta(double beta) {
    this.beta = beta;
  }
}
