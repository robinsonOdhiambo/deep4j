package com.deep4j.operations;

import org.ejml.simple.SimpleMatrix;

import static com.deep4j.utils.Matrix.assertSameShape;

public abstract class ParamOperation extends Operation {
    protected SimpleMatrix param;
    protected SimpleMatrix paramGrad;

    protected abstract SimpleMatrix paramGrad(SimpleMatrix outputGrad);

    protected ParamOperation(SimpleMatrix param) {
        this.param = param;
    }

    @Override
    public SimpleMatrix backward(SimpleMatrix outputGrad) {
        assertSameShape(this.output, outputGrad);
        this.inputGrad = this.inputGrad(outputGrad);
        this.paramGrad = this.paramGrad(outputGrad);
        assertSameShape(this.input, this.inputGrad);
        assertSameShape(this.param, this.paramGrad);

        return this.inputGrad;
    }

    public SimpleMatrix getParam() {
        return param;
    }

    public SimpleMatrix getParamGrad() {
        return paramGrad;
    }
}
