package com.deep4j.layers;

import com.deep4j.operations.Operation;
import com.deep4j.operations.ParamOperation;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;

import static com.deep4j.utils.Matrix.assertSameShape;

public abstract class Layer {
    protected int neurons;
    protected boolean first;
    protected SimpleMatrix input;
    protected SimpleMatrix output;
    protected List<SimpleMatrix> params;
    protected List<SimpleMatrix> paramGrads;
    protected List<Operation> operations;
    protected int seed;

    protected abstract void setUp(SimpleMatrix input);

    public Layer(int neurons) {
        this.neurons = neurons;
        this.first = true;
        this.params = new ArrayList<>();
        this.paramGrads = new ArrayList<>();
        this.operations = new ArrayList<>();
    }

    public SimpleMatrix forward(SimpleMatrix input) {
        if(this.first) {
            this.setUp(input);
            this.first = false;
        }
        
        this.input = input;

        for(Operation op : operations) {
            input = op.forward(input);
        }

        this.output = input;

        return this.output;
    }

    public SimpleMatrix backward(SimpleMatrix outputGrad) {
        assertSameShape(this.output, outputGrad);
        int n = operations.size();
        for(int i = n - 1; i >= 0; i--) {
            outputGrad = operations.get(i).backward(outputGrad);
        }

        SimpleMatrix inputGrad = outputGrad;
        assertSameShape(this.input, inputGrad);

        this.extractParamGrads();

        return inputGrad;
    }

    private void extractParamGrads() {
        this.paramGrads = new ArrayList<>();
        for(Operation op: operations) {
            if(op instanceof ParamOperation) {
                this.paramGrads.add(((ParamOperation)op).getParamGrad());
            }
        }
    }

    private void extractParams() {
        this.params = new ArrayList<>();
        for(Operation op: operations) {
            if(op instanceof ParamOperation) {
                this.params.add(((ParamOperation)op).getParam());
            }
        }
    }

    public List<SimpleMatrix> getParams() {
        if(params.isEmpty()) {
            extractParams();
        }
        return params;
    }

    public List<SimpleMatrix> getParamGrads() {
        return paramGrads;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    public void setFirst(boolean first) {
        this.first = first;
    }
}
