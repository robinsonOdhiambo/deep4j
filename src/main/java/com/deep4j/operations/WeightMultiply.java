package com.deep4j.operations;

import org.ejml.simple.SimpleMatrix;

public class WeightMultiply extends ParamOperation {

    public WeightMultiply(SimpleMatrix param) {
        super(param);
    }

    @Override
    protected SimpleMatrix output() {
        return this.input.mult(this.param);
    }

    @Override
    protected SimpleMatrix inputGrad(SimpleMatrix outputGrad) {
        return outputGrad.mult(this.param.transpose());
    }

    @Override
    protected SimpleMatrix paramGrad(SimpleMatrix outputGrad) {
        return this.input.transpose().mult(outputGrad);
    }
}
