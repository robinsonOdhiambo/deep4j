package com.deep4j.activations;

import com.deep4j.operations.Operation;
import org.ejml.simple.SimpleMatrix;
import static com.deep4j.utils.Matrix.elementApply;

public class Sigmoid extends Operation {
    @Override
    protected SimpleMatrix output() {
        return elementApply(this.input.negative().elementExp().plus(1.0), (r, c, denominator) -> 1.0 / denominator);
    }

    @Override
    protected SimpleMatrix inputGrad(SimpleMatrix outputGrad) {
        SimpleMatrix sigmoidBackward = elementApply(this.output, (r, c, v) -> v * (1 - v));
        return sigmoidBackward.elementMult(outputGrad);
    }
}
