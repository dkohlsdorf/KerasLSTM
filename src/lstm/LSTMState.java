package lstm;

import org.jblas.DoubleMatrix;

public class LSTMState {

    private DoubleMatrix ht, ct;

    public LSTMState(DoubleMatrix ht, DoubleMatrix ct) {
        this.ht = ht;
        this.ct = ct;
    }

    public DoubleMatrix getHt() {
        return ht;
    }

    public DoubleMatrix getCt() {
        return ct;
    }

    @Override
    public String toString() {
        StringBuilder htBuilder = new StringBuilder("ht=");
        for(int i = 0; i < ht.getColumns(); i++) {
            htBuilder.append(ht.get(0, i));
            htBuilder.append(", ");
        }

        StringBuilder ctBuilder = new StringBuilder("ct=");
        for(int i = 0; i < ct.getColumns(); i++) {
            ctBuilder.append(ct.get(0, i));
            ctBuilder.append(", ");
        }

        return "LSTMState{" +
                "\n " + htBuilder.toString() +
                "\n " + ctBuilder.toString() +
                '}';
    }
}

