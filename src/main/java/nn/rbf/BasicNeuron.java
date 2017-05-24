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

  public BasicNeuron(final ACTIVATION_FUNC activationFunc, final double output) {
    this.activationFunc = activationFunc;
    setOutput(output);
  }

  @Override
  public double calculateOutput() {
    double s = 0.0;
    for (Connection c : in) {
      s += c.getWeight() * c.getIn().getOutput();
    }

    output = activate(s);
    return output;
  }

  public void applyBackpropagation(double expectedOutput, double learningRate) {
    for (Connection c : in) {
      double ai = c.getIn().getOutput();
      switch (activationFunc) {
        case SIGMOID: {
          double partialDerivative = -(expectedOutput-output) * output*(1.0 - output) * ai;
          c.setWeight(c.getWeight() - learningRate * partialDerivative);
        }
        case LINEAR: {
          double partialDerivative = -(expectedOutput-output) * 1.0 * ai;
          c.setWeight(c.getWeight() - learningRate * partialDerivative);
        }
      }
    }
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
