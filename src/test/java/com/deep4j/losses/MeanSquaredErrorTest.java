package com.deep4j.losses;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MeanSquaredErrorTest {
    private SimpleMatrix prediction;
    private SimpleMatrix target;
    private MeanSquaredError mse;

    @BeforeEach
    void setUp() {
        mse = new MeanSquaredError(false);
        prediction = new SimpleMatrix(new DMatrixRMaj(new double[]{
                0.47143516, -1.19097569,  1.43270697, -0.3126519,
                -0.72058873, 0.88716294, 0.85958841, -0.6365235,
                0.01569637, -2.24268495
        }));

        target = new SimpleMatrix(new DMatrixRMaj(new double[]{
                1.15003572,  0.99194602,  0.95332413, -2.02125482,
                -0.33407737,  0.00211836, 0.40545341,  0.28909194,
                1.32115819, -1.54690555
        }));
    }

    @Test
    void forward() {
        setUp();
        double loss = mse.forward(prediction, target);
        assertEquals(1.2558814833316259, loss, 0.00001);
    }

    @Test
    void backward() {
        mse.forward(prediction, target);
        double[] grad = new double[]{
                -0.13572011, -0.43658434, 0.09587657,
                0.34172058, -0.07730227, 0.17700892,
                0.090827, -0.18512309, -0.26109236,
                -0.13915588
        };
        assertArrayEquals(grad, mse.backward().getDDRM().data, 0.00001);
    }
}