package com.deep4j.operations;

import org.ejml.simple.SimpleMatrix;
import static com.deep4j.utils.Matrix.assertSameShape;

public abstract class Operation {
    protected SimpleMatrix input;
    protected SimpleMatrix output;
    protected SimpleMatrix inputGrad;

    protected abstract SimpleMatrix output();
    protected abstract SimpleMatrix inputGrad(SimpleMatrix outputGrad);

    public SimpleMatrix forward(SimpleMatrix input) {
        this.input = input;
        this.output = output();
        return this.output;
    }

    public SimpleMatrix backward(SimpleMatrix outputGrad) {
        assertSameShape(this.output, outputGrad);
        this.inputGrad = this.inputGrad(outputGrad);
        assertSameShape(this.input, this.inputGrad);

        return this.inputGrad;
    }
}
