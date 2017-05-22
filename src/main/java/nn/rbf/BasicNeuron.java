package nn.rbf;

public class BasicNeuron extends Neuron {
  public enum ACTIVATION_FUNC {
    SIGMOID,
    LINEAR
  }

  private final ACTIVATION_FUNC activationFunc;

  public BasicNeuron(final ACTIVATION_FUNC activationFunc) {
    this.activationFunc = activationFunc;
  }

  public BasicNeuron(final ACTIVATION_FUNC activationFunc, final double biasOutput, final double biasWeight) {
    super(biasOutput, biasWeight);
    this.activationFunc = activationFunc;
  }

  @Override
  public double calculateOutput() {
    double s = 0.0;
    for (Connection c : in) {
      s += c.getWeight() * c.getIn().getOutput();
    }
    if (bias != null) {
      s += bias.getWeight() * biasOutput;
    }

    output = activate(s);
    return output;
  }

  private double activate(double val) {
    switch (activationFunc) {
      case LINEAR:
        return val;
      case SIGMOID:
        return sigmoid(val);
      default:
        return val;
    }
  }

  private double sigmoid(double val) {
    return 1.0 / (1.0 +  (Math.exp(-val)));
  }
}
