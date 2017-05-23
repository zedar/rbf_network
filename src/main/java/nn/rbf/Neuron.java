package nn.rbf;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public abstract class Neuron {
  protected double output = 0.0;
  protected final List<Connection> in = new ArrayList<>();

  public Neuron() {
  }

  public abstract double calculateOutput();

  public double getOutput() {
    return output;
  }

  public void setOutput(double output) {
    this.output = output;
  }

  public void addInConnections(List<? extends Neuron> neurons) {
    for (Neuron n : neurons) {
      in.add(new Connection(n, this, getRandom()));
    }
  }

  public Connection getConnection(int idx) {
    return in.get(idx);
  }

  public void addBiasConnection(Neuron bias) {
    in.add(new Connection(bias, this, getRandom()));
  }

  private double getRandom() {
    double hi = 1.0;
    double lo = -1.0;
    return (Math.random() * (hi - lo)) + lo;
  }
}
