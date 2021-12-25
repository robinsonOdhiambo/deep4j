package com.deep4j.losses;

import org.ejml.simple.SimpleMatrix;

import static com.deep4j.utils.Matrix.elementApply;

public class SoftmaxCrossEntropy extends Loss {
    protected double eps;
    protected SimpleMatrix softmax;

    public SoftmaxCrossEntropy(double eps) {
        this.eps = eps;
    }

    private double clip(double v) {
        double exp = Math.log(eps) / Math.log(10);
        double minEps;
        double maxEps;
        if(exp > 0) {
            minEps = Math.pow(10, -exp);
            maxEps = Math.pow(10, exp);
        } else {
            minEps = Math.pow(10, exp);
            maxEps = Math.pow(10, -exp);
        }

        if(v < minEps) {
            return minEps;
        }


        if(v > maxEps) {
            return -maxEps;
        }

        return v;
    }

    private SimpleMatrix softmax() {
        double[] sums = new double[prediction.numRows()];
        for(int r = 0; r <sums.length; r++) {
            sums[r] = elementApply(
                    prediction.extractVector(true, r), (row, c, v) -> Math.exp(v)
            ).elementSum();
        }

        return elementApply(prediction, (r,c,v) -> clip(Math.exp(v) / sums[r]));
    }

    @Override
    protected double output() {
        this.softmax = softmax();
        SimpleMatrix left = elementApply(softmax, (r, c, v) -> -1.0 * target.get(r, c) * Math.log(v));
        SimpleMatrix right = elementApply(softmax, (r, c, v) -> (1.0 - target.get(r, c)) * (Math.log(1-v)));
        SimpleMatrix smce = left.minus(right);
        return smce.elementSum() / prediction.numRows();
    }

    @Override
    protected SimpleMatrix inputGrad() {
        return softmax.minus(target).scale(1.0 / prediction.numRows());
    }
}
