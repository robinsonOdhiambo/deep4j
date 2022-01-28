package com.deep4j.optimizers;

import com.deep4j.Network;
import org.ejml.simple.SimpleMatrix;

import java.util.List;

public abstract class Optimizer {
    protected double learningRate;
    protected boolean first;
    protected Network net;

    protected abstract void updateRule(SimpleMatrix param, SimpleMatrix paramGrad);

    protected Optimizer(double learningRate, Network net) {
        this.learningRate = learningRate;
        this.first = true;
        this.net = net;
    }

    public void step() {
        List<List<SimpleMatrix>> params = net.params();
        List<List<SimpleMatrix>> paramGrads = net.paramGrads();
        int n = params.size();
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < params.get(i).size(); j++) {
                this.updateRule(params.get(i).get(j), paramGrads.get(i).get(j));
            }
        }
    }

    public Network getNet() {
        return net;
    }
}
