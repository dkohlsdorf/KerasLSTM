package lstm;

import org.jblas.DoubleMatrix;

import java.util.ArrayList;

public class LSTM {

    private int inSize, cellSize;
    private DoubleMatrix wf, wi, wo, wc;
    private DoubleMatrix uf, ui, uo, uc;
    private DoubleMatrix bf, bi, bo, bc;

    public LSTM(String folder, int inSize, int cellSize) {
        this.cellSize = cellSize;
        this.inSize = inSize;
        try {
            double[][] bias       = NumericHelpers.readMatrix(folder + "/lstm_bias.txt",      4 * cellSize, 1);
            double[][] weights    = NumericHelpers.readMatrix(folder + "/lstm_weights.txt",   inSize,   4 * cellSize);
            double[][] recurrent  = NumericHelpers.readMatrix(folder + "/lstm_recurrent.txt", cellSize, 4 * cellSize);
            parseWeights(weights);
            parseBias(bias);
            parseRecurrent(recurrent);
        } catch (Exception e) {
            System.err.println("Can not load weights in folder: " + folder);
            System.exit(0);
        }
    }

    private void parseWeights(double[][] matrix) {
        wi = NumericHelpers.colSlice(matrix, 0           ,     cellSize);
        wf = NumericHelpers.colSlice(matrix,     cellSize, 2 * cellSize);
        wc = NumericHelpers.colSlice(matrix, 2 * cellSize, 3 * cellSize);
        wo = NumericHelpers.colSlice(matrix, 3 * cellSize, 4 * cellSize);
    }

    private void parseRecurrent(double[][] matrix) {
        ui = NumericHelpers.colSlice(matrix, 0           ,     cellSize);
        uf = NumericHelpers.colSlice(matrix,     cellSize, 2 * cellSize);
        uc = NumericHelpers.colSlice(matrix, 2 * cellSize, 3 * cellSize);
        uo = NumericHelpers.colSlice(matrix, 3 * cellSize, 4 * cellSize);
    }

    private void parseBias(double[][] matrix) {
        bi = NumericHelpers.rowSlice(matrix, 0           ,     cellSize);
        bf = NumericHelpers.rowSlice(matrix,     cellSize, 2 * cellSize);
        bc = NumericHelpers.rowSlice(matrix, 2 * cellSize, 3 * cellSize);
        bo = NumericHelpers.rowSlice(matrix, 3 * cellSize, 4 * cellSize);
    }

    public LSTMState next(DoubleMatrix input, LSTMState state) {
        DoubleMatrix ft = NumericHelpers.sigmoid(input.mmul(wf).add(state.getHt().mmul(uf)).add(bf));
        DoubleMatrix it = NumericHelpers.sigmoid(input.mmul(wi).add(state.getHt().mmul(ui)).add(bi));
        DoubleMatrix ot = NumericHelpers.sigmoid(input.mmul(wo).add(state.getHt().mmul(uo)).add(bo));
        DoubleMatrix ct = NumericHelpers.tanh(input.mmul(wc).add(state.getHt().mmul(uc)).add(bc));
        ct = ft.mul(state.getCt()).add(it.mul(ct));
        DoubleMatrix ht = ot.mul(NumericHelpers.tanh(ct));
        return new LSTMState(ht, ct);
    }

    public ArrayList<LSTMState> decode(double[][] seq) {
        ArrayList<LSTMState> states = new ArrayList<>();
        LSTMState state = new LSTMState(new DoubleMatrix(new double[1][cellSize]), new DoubleMatrix(new double[1][cellSize]));
        for(int i = 0; i < seq.length; i++) {
            state = next(new DoubleMatrix(new double[][]{ seq[i] }), state);
            states.add(state);
        }
        return states;
    }

    public static void main(String ... args) throws Exception {
        double[][] expected   = NumericHelpers.readMatrix("model_weights/output.txt", 1000, 100);
        double[][] input      = NumericHelpers.readMatrix("model_weights/input.txt",  1000,  10);
        LSTM lstm = new LSTM("model_weights", 10, 100);
        ArrayList<LSTMState> states = lstm.decode(input);
        for(int i = 0; i < expected.length; i++) {
            DoubleMatrix x = states.get(i).getHt().sub(new DoubleMatrix(new double[][]{ expected[i] }).transpose());
            double error = 0.0;
            for(int j = 0; j < x.getColumns(); j++) {
                error += Math.pow(x.get(0, j), 2.0);
            }
            System.out.println(error);
        }
    }

}
