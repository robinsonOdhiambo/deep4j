package com.deep4j.optimizers;

import com.deep4j.Network;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

import static com.deep4j.utils.Matrix.elementApply;

public class SGDMomentum extends Optimizer {
    private double momentum = 0.9;
    private final List<List<SimpleMatrix>> velocities = new ArrayList<>();

    public SGDMomentum(double learningRate, double momentum, Network net) {
        super(learningRate, net);
        this.momentum = momentum;
    }

    @Override
    public void step() {
        if(first) {
            for(List<SimpleMatrix> params: net.params()) {
                List<SimpleMatrix> layerVelocities = new ArrayList<>();
                for(SimpleMatrix param: params) {
                    SimpleMatrix velocity = new SimpleMatrix(param.numRows(), param.numCols());
                    velocity.fill(0);
                    layerVelocities.add(velocity);
                }
                velocities.add(layerVelocities);
            }
            first = false;
        }

        List<List<SimpleMatrix>> params = net.params();
        List<List<SimpleMatrix>> paramGrads = net.paramGrads();
        int n = params.size();
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < params.get(i).size(); j++) {
                this.updateRule(params.get(i).get(j), paramGrads.get(i).get(j), velocities.get(i).get(j));
            }
        }
    }

    @Override
    protected void updateRule(SimpleMatrix param, SimpleMatrix paramGrad) {
        // not implemented
        throw new RuntimeException("Method not implemented");
    }

    protected void updateRule(SimpleMatrix param,
                              SimpleMatrix paramGrad,
                              SimpleMatrix velocity) {
        elementApply(paramGrad, (r, c, grad) -> {
            velocity.set(r, c, momentum * velocity.get(r, c) + learningRate * grad);
            return grad;
        });

        elementApply(velocity, (r, c, v) -> {
            param.set(r, c, param.get(r, c) - v);
            return v;
        });
    }
}
