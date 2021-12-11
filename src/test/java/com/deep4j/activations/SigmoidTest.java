package com.deep4j.activations;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SigmoidTest {
    private Sigmoid sigmoid;
    private SimpleMatrix input;

    @BeforeEach
    void setUp() {
        sigmoid = new Sigmoid();
        input = new SimpleMatrix(new DMatrixRMaj(new double[]{
                0.47143516, -1.19097569,  1.43270697, -0.3126519,
                -0.72058873, 0.88716294, 0.85958841, -0.6365235,
                0.01569637, -2.24268495
        }));
    }

    @Test
    void forward() {
        double[] expected = new double[]{
                0.61572338, 0.23308448, 0.80732274, 0.42246757,
                0.32726335, 0.70830436, 0.70257466, 0.34603283,
                0.50392401, 0.09598232
        };
        assertArrayEquals(expected, sigmoid.forward(input).getDDRM().data, 0.00001);
    }

    @Test
    void backward() {
        SimpleMatrix outputGrad = new SimpleMatrix(new DMatrixRMaj(new double[] {
                1.15003572,  0.99194602,  0.95332413, -2.02125482, -0.33407737,
                0.00211836, 0.40545341,  0.28909194,  1.32115819, -1.54690555
        }));

        double[] expected = new double[] {
                2.72107766e-01,  1.77316407e-01,  1.48292173e-01, -4.93163382e-01,
                -7.35511580e-02,  4.37673834e-04,  8.47249677e-02,  6.54198035e-02,
                3.30269205e-01, -1.34224548e-01
        };

        sigmoid.forward(input);
        assertArrayEquals(expected, sigmoid.backward(outputGrad).getDDRM().data, 0.00001);
    }
}