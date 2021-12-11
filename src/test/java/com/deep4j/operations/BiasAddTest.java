package com.deep4j.operations;

import org.ejml.simple.SimpleMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BiasAddTest {
    private BiasAdd biasAdd;
    private SimpleMatrix input;


    @BeforeEach
    void setUp() {
        input = new SimpleMatrix(new double[][]{
                new double[] {4.71435164e-01, -1.19097569e+00}, new double[]{1.43270697e+00, -3.12651896e-01},
                new double[]{-7.20588733e-01, 8.87162940e-01}, new double[]{8.59588414e-01, -6.36523504e-01},
                new double[]{1.56963721e-02, -2.24268495e+00}, new double[]{1.15003572e+00,  9.91946022e-01},
                new double[]{9.53324128e-01, -2.02125482e+00}, new double[]{-3.34077366e-01,  2.11836468e-03},
                new double[]{4.05453412e-01, 2.89091941e-01},new double[]{1.32115819e+00, -1.54690555e+00}
        });

        SimpleMatrix param = new SimpleMatrix(new double[][]{{-0.20264632, -0.65596934}});
        biasAdd = new BiasAdd(param);
    }

    @Test
    void forward() {
        double[] expected = new double[]{
                0.26878884, -1.84694504, 1.23006064, -0.96862124, -0.92323506, 0.2311936,
                0.65694209, -1.29249285, -0.18694995, -2.8986543, 0.9473894, 0.33597668,
                0.7506778, -2.67722416, -0.53672369, -0.65385098, 0.20280709, -0.3668774,
                1.11851187, -2.2028749
        };

        assertArrayEquals(expected, biasAdd.forward(input).getDDRM().data, 0.00001);
    }

    @Test
    void backward() {
        SimpleMatrix grad = new SimpleMatrix(new double[][]{
                new double[] {0.19342138, 0.55343891}, new double[] {1.31815155, -0.46930528},
                new double[] {0.67555409, -1.81702723}, new double[] {-0.18310854, 1.05896919},
                new double[] {-0.39784023, 0.33743765}, new double[] {1.04757857, 1.04593826},
                new double[] {0.86371729, -0.12209157}, new double[] {0.12471295, -0.32279481},
                new double[] {0.84167471, 2.39096052}, new double[] {0.07619959, -0.56644593}
        });
        double[] expected = new double[]{
                0.19342138, 0.55343891, 1.31815155, -0.46930528,
                0.67555409, -1.81702723, -0.18310854, 1.05896919,
                -0.39784023, 0.33743765, 1.04757857,  1.04593826,
                0.86371729, -0.12209157, 0.12471295, -0.32279481,
                0.84167471, 2.39096052, 0.07619959, -0.56644593
        };

        biasAdd.forward(input);
        assertArrayEquals(expected, biasAdd.backward(grad).getDDRM().data, 0.00001);
    }
}