package nn.rbf.utils;

public class DataSetUtils {
  public static void normalizeInRangeZeroOne(double[][] ds) {
    double min = 0.0;
    double max = 0.0;
    for (int i = 0; i < ds.length; i++) {
      double d = ds[i][0];
      if (d < min) {
        min = d;
      }
      if (d > max) {
        max = d;
      }
    }
    System.out.printf("NORMALIZE: MIN: %f, MAX: %f\n", min, max);
    for (int i = 0; i < ds.length; i++) {
      //data[i][0] = (data[i][0] - min) / (max - min);
      // (b-a)(x-min)/(max-min) + a
      double a = 0.0;
      double b = 1.0;
      ds[i][0] = (b-a)*(ds[i][0] - min) / (max - min) + a;
    }
  }

  public static void printDS(String prompt, double[][] ds, Integer head) {
    System.out.println("-------------------------");
    System.out.println(prompt);
    if (head == null) head = ds.length;
    for (int i=0; i<ds.length && i<head; i++) {
      StringBuilder sb = new StringBuilder();
      sb.append(i).append(" : ");
      for (int j=0; j<ds[i].length; j++) {
        sb.append(ds[i][j]).append(" ");
      }
      System.out.println(sb.toString());
    }
  }

}
