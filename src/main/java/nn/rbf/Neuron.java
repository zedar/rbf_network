package nn.rbf;

import java.util.ArrayList;
import java.util.List;

public abstract class Neuron {
  private double output = 0.0;
  private final List<Connection> in = new ArrayList<>();
  private final Connection bias;

  public Neuron() {
    this.bias = new Connection(null, null, 0.0);
  }

  public Neuron(double bias) {
    this.bias = new Connection(null, null, bias);
  }

  public abstract double cakculateOutput();
}
