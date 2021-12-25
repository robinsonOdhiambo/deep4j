package com.deep4j.losses;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SoftmaxCrossEntropyTest {
    private SimpleMatrix pred;
    private SimpleMatrix target;
    private SoftmaxCrossEntropy sce;

    @BeforeEach
    void setUp() {
        pred = new SimpleMatrix(new double[][] {
                {5, 3, 2}
        });

        target = new SimpleMatrix(new double[][] {
                {0, 0, 1}
        });

        sce = new SoftmaxCrossEntropy(1e-9);
    }

    @Test
    void forward() {
        assertEquals(5.167991105, sce.forward(pred, target), 0.05);
    }

    @Test
    void backward() {
        sce.forward(pred, target);
        assertArrayEquals(new double[]{0.84, 0.11, -0.96}, sce.backward().getDDRM().data, 0.005);
    }
}