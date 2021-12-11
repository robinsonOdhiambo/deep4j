package com.deep4j.optimizers;

import com.deep4j.Network;
import org.ejml.simple.SimpleMatrix;

import static com.deep4j.utils.Matrix.elementApply;

public class SGD extends Optimizer {

    public SGD(double learningRate, Network net) {
        super(learningRate, net);
    }

    @Override
    protected void updateRule(SimpleMatrix param, SimpleMatrix paramGrad) {
        elementApply(paramGrad, (r, c, v) -> {
            param.set(r, c, param.get(r, c) - learningRate * v);
            return v;
        });
    }
}
