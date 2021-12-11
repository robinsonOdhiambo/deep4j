package com.deep4j.layers;

import com.deep4j.operations.BiasAdd;
import com.deep4j.operations.Operation;
import com.deep4j.operations.WeightMultiply;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static com.deep4j.utils.Matrix.elementApply;

public class Dense extends Layer {
    private final Operation activation;
    private final Random random;
    private final boolean paramInit;

    public Dense(int neurons, int seed, Operation activation) {
        super(neurons);
        paramInit = false;
        this.activation = activation;
        this.random = new Random(seed);
    }

    public Dense(int neurons, List<SimpleMatrix> params, Operation activation) {
        super(neurons);
        paramInit = true;
        this.params = params;
        this.activation = activation;
        this.random = new Random();
    }

    private SimpleMatrix genMatrix(int r, int c) {
        return elementApply(new SimpleMatrix(r, c), (row, col, v) -> random.nextGaussian());
    }

    @Override
    protected void setUp(SimpleMatrix input) {
        int numOfInputs = input.numCols();
        if(!paramInit) {
            this.params = Arrays.asList(
                    genMatrix(numOfInputs, this.neurons),
                    genMatrix(1, this.neurons)
            );
        }

        this.operations = Arrays.asList(
                new WeightMultiply(this.params.get(0)),
                new BiasAdd(this.params.get(1)),
                this.activation
        );
    }
}