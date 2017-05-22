package nn.rbf;

public class Connection {
  private final Neuron in;
  private final Neuron out;
  private double weight = 0.0;

  public Connection(final Neuron in, final Neuron out) {
    this.in = in;
    this.out = out;
  }

  public Connection(final Neuron in, final Neuron out, final double weight) {
    this.in = in;
    this.out = out;
    this.weight = weight;
  }

  public double getWeight() {
    return weight;
  }

  public void setWeight(double weight) {
    this.weight = weight;
  }

  public Neuron getIn() {
    return in;
  }

  public Neuron getOut() {
    return out;
  }
}
