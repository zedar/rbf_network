package nn.rbf;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class KMeans {
  private static class Cluster {
    double[] mu;
    double[] tempmu;
    int neighbourhood = 0;
    private boolean updatedTempMe = false;
    private boolean applyTempMu = false;

    Cluster(final double[] mu) {
      this.mu = mu;
      tempmu = new double[this.mu.length];
      reset();
    }

    Cluster(final double[] mu, final double[] out) {
      this.mu = new double[mu.length+out.length];
      for (int i = 0; i < mu.length; i++) {
        this.mu[i] = mu[i];
      }
      for (int i = 0; i < out.length; i++) {
        this.mu[i+mu.length] = out[i];
      }
      tempmu = new double[this.mu.length];
      reset();
    }

    Cluster updateMu() {
      if (!updatedTempMe) {
        updateTempMu();
      }
      if (!applyTempMu) {
        return this;
      }
      for (int i = 0; i < mu.length; i++) {
        mu[i] = tempmu[i];
      }
      reset();
      return this;
    }

    Cluster updateTempMu() {
      if (updatedTempMe) {
        return this;
      }
      updatedTempMe = true;
      if (neighbourhood == 0) {
        for (int i = 0; i < mu.length; i++) {
          tempmu[i] = mu[i];
        }
        return this;
      }
      for (int i = 0; i < tempmu.length; i++) {
        tempmu[i] = tempmu[i] / neighbourhood;
      }
      applyTempMu = true;
      return this;
    }

    Cluster reset() {
      for (int i = 0; i < mu.length; i++) {
        tempmu[i] = 0.0;
      }
      neighbourhood = 0;
      applyTempMu = false;
      updatedTempMe = false;
      return this;
    }

    Cluster assign(double[] observation) {
      for (int i = 0; i < tempmu.length; i++) {
        tempmu[i] = tempmu[i] + observation[i];
      }
      neighbourhood++;
      return this;
    }
  }

  public static void classify(final double[][] trainin, final List<RadialNeuron> centers, final double maxError, final int maxIterations) {
    // init clusters
    List<Cluster> clusters = new ArrayList<>();
    for (RadialNeuron rn : centers) {
      clusters.add(new Cluster(rn.getMu()));
    }
    for (int i = 0; i < maxIterations; i++) {
      for (int j = 0; j < trainin.length; j++) {
        double mindist = 0.0;
        int bestclusteridx = -1;
        for (int k = 0; k < clusters.size(); k++) {
          Cluster c = clusters.get(k);
          double dist = EuclideanDistance.calc(c.mu, trainin[j]);
          if (bestclusteridx == -1 || dist < mindist) {
            mindist = dist;
            bestclusteridx = k;
          }
        }
        clusters.get(bestclusteridx).assign(trainin[j]);
      }
      double dist = 0.0;
      for (int j = 0; j < clusters.size(); j++) {
        Cluster c = clusters.get(j);
        c.updateTempMu();
        dist += EuclideanDistance.calc(c.mu, c.tempmu);
        c.updateMu();
      }

      if (dist <= maxError) {
        break;
      }
      if (i%1000 == 0) {
        System.out.printf("K-MEANS EPOCH: %d, DISTANCE: %f\n", i, dist);
      }
    }
    // Update radial neurons
    for (int i = 0; i < centers.size(); i++) {
      Cluster c = clusters.get(i);
      centers.get(i).setMu(c.mu);
    }
  }

  public static void classify2(int numOfClasters, final double[][] trainin, final double[][] trainout, final List<RadialNeuron> centers, final double maxError, final int maxIterations, boolean print) {
    Random rand = new Random();
    double[][] trainDS = merge(trainin, trainout);

    PrintWriter fClustersTrain = null;
    PrintWriter fClustersInit = null;
    PrintWriter fClustersOut = null;
    try {
      if (print) {
        fClustersTrain = new PrintWriter(new FileWriter("out/plot_clusters_train.txt"));
        fClustersInit = new PrintWriter(new FileWriter("out/plot_clusters_init.txt"));
        fClustersOut = new PrintWriter(new FileWriter("out/plot_clusters_out.txt"));
        if (trainDS[0].length == 2) {
          for (int i = 0; i < trainDS.length; i++) {
            fClustersTrain.println("\t" + trainDS[i][0] + "\t" + trainDS[i][1]);
          }
        }
      }
      // init centers

      List<Cluster> clusters = new ArrayList<>();
      for (int i = 0; i < numOfClasters; i++) {
        int pos = rand.nextInt(trainin.length);
        clusters.add(new Cluster(trainDS[pos]));
      }

      if (print && clusters.get(0).mu.length == 2) {
        for (int i = 0; i < clusters.size(); i++) {
          fClustersInit.println("\t" + clusters.get(i).mu[0] + "\t" + clusters.get(i).mu[1]);
        }
      }

      for (int i = 0; i < maxIterations; i++) {
        for (int j = 0; j < trainDS.length; j++) {
          double mindist = 0.0;
          int bestclusteridx = -1;
          for (int k = 0; k < clusters.size(); k++) {
            Cluster c = clusters.get(k);
            double dist = EuclideanDistance.calc(c.mu, trainDS[j]);
            if (bestclusteridx == -1 || dist < mindist) {
              mindist = dist;
              bestclusteridx = k;
            }
          }
          clusters.get(bestclusteridx).assign(trainDS[j]);
        }
        double dist = 0.0;
        for (int j = 0; j < clusters.size(); j++) {
          Cluster c = clusters.get(j);
          c.updateTempMu();
          dist += EuclideanDistance.calc(c.mu, c.tempmu);
          c.updateMu();
        }

        if (dist <= maxError) {
          System.out.printf("K-MEANS EPOCH: %d, DISTANCE: %f\n", i, dist);
          break;
        }
        if (i % 1000 == 0) {
          System.out.printf("K-MEANS EPOCH: %d, DISTANCE: %f\n", i, dist);
        }

      }

      if (print && clusters.get(0).mu.length == 2) {
        for (int i = 0; i < clusters.size(); i++) {
          fClustersOut.println("\t" + clusters.get(i).mu[0] + "\t" + clusters.get(i).mu[1]);
        }
      }

      // Update radial neurons
      for (int i = 0; i < centers.size(); i++) {
        Cluster c = clusters.get(i);
        double[] mu = new double[trainin[0].length];
        for (int j = 0; j < mu.length; j++) {
          mu[j] = c.mu[j];
        }
        centers.get(i).setMu(mu);
      }
    } catch (Exception ex) {
      ex.printStackTrace();
    } finally {
      if (fClustersInit != null) {
        fClustersInit.close();
      }
      if (fClustersOut != null) {
        fClustersOut.close();
      }
      if (fClustersTrain != null) {
        fClustersTrain.close();
      }
    }
  }

  private static double[][] merge(double[][] trainin, double[][] trainout) {
    double[][] out = new double[trainin.length][];
    for (int i = 0; i < trainin.length; i++) {
      out[i] = new double[trainin[i].length + trainout[i].length];
      for (int j = 0; j < trainin[i].length; j++) {
        out[i][j] = trainin[i][j];
      }
      for (int j = 0; j < trainout[i].length; j++) {
        out[i][j+trainin[i].length] = trainout[i][j];
      }
    }
    return out;
  }
}
