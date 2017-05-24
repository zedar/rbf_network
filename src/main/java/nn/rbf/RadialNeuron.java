package nn.rbf;

public class RadialNeuron extends Neuron {
  private double[] mu;
  private double beta;
  private double radius;

  public RadialNeuron(final double beta) {
    this.beta = beta;
    this.beta = 5.0;
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

    //output = Math.exp(-beta * s /*/ (2.0*beta*beta)*/);
    output = Math.exp(-s / (radius*radius));
    return output;
  }

  public void applyBackPropagation(double calculatedOutput, double expectedOutput, double weight, double learningRate) {
    double phi = output;
    double diff = expectedOutput - calculatedOutput;
    for (int i = 0; i < mu.length; i++) {
      mu[i] = mu[i] + learningRate * diff * weight * phi * (in.get(i).getIn().getOutput() - mu[i])/(beta*beta);//(beta*beta);
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

  public double getRadius() {
    return radius;
  }

  public void setRadius(double radius) {
    this.radius = radius;
  }
}
