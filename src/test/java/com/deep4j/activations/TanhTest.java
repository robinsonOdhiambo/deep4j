package com.deep4j.activations;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TanhTest {
    private Tanh tanh;
    private SimpleMatrix input;

    @BeforeEach
    void setUp() {
        tanh = new Tanh();
        input = new SimpleMatrix(new DMatrixRMaj(new double[]{
                0.47143516, -1.19097569,  1.43270697, -0.3126519,
                -0.72058873, 0.88716294, 0.85958841, -0.6365235,
                0.01569637, -2.24268495
        }));
    }

    @Test
    void forward() {
        double[] expected = new double[] {
                0.43935817, -0.83088122,  0.89222, -0.3028477,
                -0.61727385, 0.70998962, 0.69604555, -0.5625278,
                0.01569508, -0.97770588
        };

        assertArrayEquals(expected, tanh.forward(input).getDDRM().data, 0.00001);
    }

    @Test
    void backward() {
        SimpleMatrix outputGrad = new SimpleMatrix(new DMatrixRMaj(new double[] {
                1.15003572,  0.99194602,  0.95332413, -2.02125482,
                -0.33407737,  0.00211836, 0.40545341,  0.28909194,
                1.32115819, -1.54690555
        }));

        double[] expected = new double[] {
                9.28037886e-01,  3.07142587e-01,  1.94424227e-01, -1.83587194e+00,
                -2.06784869e-01,  1.05052826e-03,  2.09019582e-01,  1.97612403e-01,
                1.32083274e+00, -6.82049364e-02
        };

        tanh.forward(input);
        assertArrayEquals(expected, tanh.backward(outputGrad).getDDRM().data, 0.00001);
    }
}