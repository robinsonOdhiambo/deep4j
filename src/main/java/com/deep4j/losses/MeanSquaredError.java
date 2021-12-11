package com.deep4j.losses;

import org.ejml.simple.SimpleMatrix;

import static com.deep4j.utils.Matrix.elementApply;
import static com.deep4j.utils.Matrix.sumCol;

public class MeanSquaredError extends Loss {
    private final boolean normalize;

    public MeanSquaredError(boolean normalize) {
        this.normalize = normalize;
    }

    @Override
    protected double output() {
        if(this.normalize) {
            this.prediction = elementApply(sumCol(this.prediction), (r, c, v) -> prediction.get(r, c) / v);
        }

        return elementApply(
                prediction.minus(target),
                (r, c, v) -> Math.pow(v, 2)
        ).elementSum() / prediction.numRows();
    }

    @Override
    protected SimpleMatrix inputGrad() {
        return prediction.minus(target).scale(2.0 / prediction.numRows());
    }
}
