package com.deep4j;

import com.deep4j.layers.Layer;
import com.deep4j.losses.Loss;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

public class Network {
    protected List<Layer> layers;
    protected int seed;
    protected Loss loss;

    public Network(List<Layer> layers, int seed, Loss loss) {
        this.layers = layers;
        this.seed = seed;
        this.loss = loss;

        for(Layer layer: layers) {
            layer.setSeed(seed);
        }
    }

    public SimpleMatrix forward(SimpleMatrix obs) {
        SimpleMatrix obsOut = obs;

        for(Layer layer: layers) {
            obsOut = layer.forward(obsOut);
        }

        return obsOut;
    }

    public double forwardLoss(SimpleMatrix obs, SimpleMatrix target) {
        SimpleMatrix prediction = this.forward(obs);
        return this.loss.forward(prediction, target);
    }

    public double trainBatch(SimpleMatrix obs, SimpleMatrix target) {
        SimpleMatrix prediction = this.forward(obs);
        double batchLoss = this.loss.forward(prediction, target);
        SimpleMatrix lossGrad = this.loss.backward();
        this.backward(lossGrad);

        return batchLoss;
    }

    public void backward(SimpleMatrix lossGrad) {
        SimpleMatrix grad = lossGrad;
        int n = layers.size();
        for(int i = n - 1; i >= 0; i--) {
            grad = layers.get(i).backward(grad);
        }

//        return grad;
    }

    public List<List<SimpleMatrix>> params() {
        List<List<SimpleMatrix>> params = new ArrayList<>();
        for(Layer layer: layers) {
            params.add(layer.getParams());
        }

        return params;
    }

    public List<List<SimpleMatrix>> paramGrads() {
        List<List<SimpleMatrix>> paramGrads = new ArrayList<>();
        for(Layer layer: layers) {
            paramGrads.add(layer.getParamGrads());
        }

        return paramGrads;
    }

    public Network copy() {
        return this; // TODO implement deep copy
    }
}
