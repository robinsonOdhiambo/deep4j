package com.deep4j.operations;

import org.ejml.simple.SimpleMatrix;

import static com.deep4j.utils.Matrix.elementApply;
import static com.deep4j.utils.Matrix.sumRow;

public class BiasAdd extends ParamOperation {

    public BiasAdd(SimpleMatrix param) {
        super(param);
    }
    @Override
    protected SimpleMatrix output() {
        SimpleMatrix param = elementApply(this.input.createLike(), (r, c, v) -> this.param.get(0, c));
        return this.input.plus(param);
    }

    @Override
    protected SimpleMatrix inputGrad(SimpleMatrix outputGrad) {
        SimpleMatrix ones = new SimpleMatrix(
                this.input.numRows(),
                this.input.numCols()
        );
        ones.fill(1.0);
        return ones.elementMult(outputGrad);
    }

    @Override
    protected SimpleMatrix paramGrad(SimpleMatrix outputGrad) {
        SimpleMatrix ones = new SimpleMatrix(
                outputGrad.numRows(),
                this.param.numCols()
        );
        ones.fill(1.0);
        return sumRow(ones.elementMult(outputGrad));
    }
}
