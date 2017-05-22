package nn.rbf;

import java.util.ArrayList;
import java.util.List;

public abstract class Neuron {
  protected double output = 0.0;
  protected double biasOutput = -1.0;
  protected final List<Connection> in = new ArrayList<>();
  protected final Connection bias;

  public Neuron() {
    this.bias = new Connection(null, null, 0.0);
  }

  public Neuron(final double biasOutput, final double biasWeight) {
    this.biasOutput = biasOutput;
    this.bias = new Connection(null, null, biasWeight);
  }

  public abstract double cakculateOutput();

  public double getOutput() {
    return output;
  }

  public void setOutput(double output) {
    this.output = output;
  }

  public double getBiasOutput() {
    return biasOutput;
  }

  public void setBiasOutput(double biasOutput) {
    this.biasOutput = biasOutput;
  }
}
