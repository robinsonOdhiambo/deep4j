package com.deep4j.optimizers;

import com.deep4j.Network;
import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

@ExtendWith(MockitoExtension.class)
class SGDTest {
    private SGD sgd;
    @Mock
    private Network network;
    private SimpleMatrix param;

    @BeforeEach
    void setUp() {
        param = new SimpleMatrix(new DMatrixRMaj(new double[]{
                 0.47143516, -1.19097569, 1.43270697, -0.3126519 ,
                -0.72058873, 0.88716294, 0.85958841, -0.6365235,
                 0.01569637, -2.24268495
        }));

        sgd = new SGD(0.01, network);
    }

    @Test
    void step() {
        SimpleMatrix grad = new SimpleMatrix(new DMatrixRMaj(new double[]{
                1.15003572, 0.99194602, 0.95332413, -2.02125482,
                -0.33407737, 0.00211836, 0.40545341, 0.28909194,
                1.32115819, -1.54690555
        }));

         double[] expected = new double[] {
                  0.45993481, -1.20089515, 1.42317373, -0.29243935,
                 -0.71724796, 0.88714176, 0.85553388, -0.63941442,
                 0.00248479, -2.2272159
         };

        sgd.updateRule(param, grad);

        assertArrayEquals(expected, param.getDDRM().data, 0.00001);
    }
}