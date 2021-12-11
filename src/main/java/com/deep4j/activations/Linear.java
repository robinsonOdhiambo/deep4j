package com.deep4j.activations;

import com.deep4j.operations.Operation;
import org.ejml.simple.SimpleMatrix;

public class Linear extends Operation {
    @Override
    protected SimpleMatrix output() {
        return this.input;
    }

    @Override
    protected SimpleMatrix inputGrad(SimpleMatrix outputGrad) {
        return outputGrad;
    }
}
