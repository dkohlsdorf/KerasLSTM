package lstm;

import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.FileReader;

public class NumericHelpers {

    private static final double SLOPE = 0.2;
    private static final double SHIFT = 0.5;

    public static DoubleMatrix colSlice(double[][] matrix, int sliceStart, int sliceStop) {
        int rows = matrix.length;
        double[][] slice = new double[rows][sliceStop - sliceStart];
        for(int i = 0; i < rows; i++) {
            int pos = 0;
            for(int j = sliceStart; j < sliceStop; j++, pos++) {
                slice[i][pos] = matrix[i][j];
            }
        }
        return new DoubleMatrix(slice);
    }

    public static DoubleMatrix rowSlice(double[][] matrix, int sliceStart, int sliceStop) {
        int cols = matrix[0].length;
        double[][] slice = new double[sliceStop - sliceStart][cols];
        int pos = 0;
        for(int i = sliceStart; i < sliceStop; i++, pos++) {
            for(int j = 0; j < cols; j++) {
                slice[pos][j] = matrix[i][j];
            }
        }
        return new DoubleMatrix(slice);
    }

    public static DoubleMatrix sigmoid(DoubleMatrix matrix) {
        for(int i = 0; i < matrix.getRows(); i++) {
            for(int j = 0; j < matrix.getColumns(); j++) {
                double x = matrix.get(i, j) * SLOPE + SHIFT;
                if (x > 1.0) {
                    matrix.put(i, j, 1);
                } else if (x < 0.0) {
                    matrix.put(i, j, 0);
                } else {
                    matrix.put(i, j, x);
                }
            }
        }
        return matrix;
    }

    public static DoubleMatrix tanh(DoubleMatrix matrix) {
        for(int i = 0; i < matrix.getRows(); i++) {
            for(int j = 0; j < matrix.getColumns(); j++) {
                double eNeg = Math.exp(-matrix.get(i, j));
                double ePos = Math.exp(matrix.get(i, j));
                double tanh = (ePos - eNeg) / (ePos + eNeg);
                matrix.put(i, j, tanh);
            }
        }
        return matrix;
    }

    public static double[][] readMatrix(String file, int rows, int cols) throws Exception {
        double[][] mat = new double[rows][cols];
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        int i = 0;
        while ((line = reader.readLine()) != null) {
            String cmp[] = line.split(" ");
            for(int j = 0; j < cols; j++) {
                mat[i][j] = Double.parseDouble(cmp[j]);
            }
            i++;
        }
        reader.close();
        return mat;
    }

}
