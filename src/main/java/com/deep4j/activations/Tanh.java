package com.deep4j.activations;

import com.deep4j.operations.Operation;
import org.ejml.simple.SimpleMatrix;
import static com.deep4j.utils.Matrix.elementApply;

public class Tanh extends Operation {
    @Override
    protected SimpleMatrix output() {
        return elementApply(this.input, (r, c, v) -> Math.tanh(v));
    }

    @Override
    protected SimpleMatrix inputGrad(SimpleMatrix outputGrad) {
        return outputGrad.elementMult(elementApply(this.output, (r, c, v) -> (1 - v * v)));
    }
}
