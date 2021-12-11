package com.deep4j.losses;

import org.ejml.simple.SimpleMatrix;

import static com.deep4j.utils.Matrix.assertSameShape;

public abstract class Loss {
    protected SimpleMatrix prediction;
    protected SimpleMatrix target;
    protected SimpleMatrix inputGrad;

    protected abstract double output();
    protected abstract SimpleMatrix inputGrad();

    public double forward(SimpleMatrix prediction, SimpleMatrix target) {
        assertSameShape(prediction, target);
        this.prediction = prediction;
        this.target = target;

        return this.output();
    }

    public SimpleMatrix backward() {
        this.inputGrad = this.inputGrad();
        assertSameShape(this.prediction, this.inputGrad);

        return this.inputGrad;
    }
}
