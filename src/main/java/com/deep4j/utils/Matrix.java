package com.deep4j.utils;


import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public final class Matrix {
    public static void assertSameShape(SimpleMatrix A, SimpleMatrix B) {
        if (A.numCols() != B.numCols() || A.numRows() != B.numRows()) {
            String errorMsg = String.format("Two tensors should have the same shame instead, " +
                    "The first tensor's shape is (%d, %d) and the second tensor's is (%d, %d).",
                    A.numRows(), A.numCols(), B.numRows(), B.numCols());
            throw new RuntimeException(errorMsg);
        }
    }

    public static SimpleMatrix sumRow(SimpleMatrix mat) {
        int r = mat.numRows();
        int c = mat.numCols();
        DMatrixRMaj res = new DMatrixRMaj(1, c);
        res.fill(0);

        for(int i = 0; i < c; i++) {
            for(int j = 0; j < r; j++) {
                res.set(0, i, res.get(0, i) + mat.get(j, i));
            }
        }

        return new SimpleMatrix(res);
    }

    public static SimpleMatrix sumCol(SimpleMatrix mat) {
        int r = mat.numRows();
        int c = mat.numCols();
        DMatrixRMaj res = new DMatrixRMaj(r, 1);
        res.fill(0);

        for(int i = 0; i < r; i++) {
            for(int j = 0; j < c; j++) {
                res.add(0, i, res.get(0, i) + mat.get(j, i));
            }
        }

        return new SimpleMatrix(res);
    }

    public interface Func {
        double apply(int r, int c, double v);
    }

    public static SimpleMatrix elementApply(SimpleMatrix mat, Func func) {
        SimpleMatrix output = new SimpleMatrix(mat);
        for(int r = 0; r < output.numRows(); r++) {
            for(int c = 0; c < output.numCols(); c++) {
                output.set(r, c, func.apply(r, c, mat.get(r, c)));
            }
        }

        return output;
    }

    private static void replaceRow(SimpleMatrix a, int row, SimpleMatrix rowVec) {
        if(a.numCols() != rowVec.numCols()) {
            throw new IllegalArgumentException(
                    "For row replacement to work the two matrices must have the same number of columns"
            );
        }
        int cols = a.numCols();
        for(int c = 0; c < cols; c++) {
            a.setRow(row, c, rowVec.get(c));
        }
    }

    private static void swapRows(SimpleMatrix a, int i, int j) {
        SimpleMatrix temp = a.extractVector(true, i).copy();
        replaceRow(a, i, a.extractVector(true, j));
        replaceRow(a, j, temp);
    }

    public static void permute(SimpleMatrix a, SimpleMatrix b) {
        if(a.numRows() != b.numRows()) {
            throw new IllegalArgumentException(
                    "For permutation to work the two matrices must have the same number of rows"
            );
        }
        int n = a.numRows();
        List<Integer> list = new ArrayList<>();
        for(int i = 0; i < n; i++) {
            list.add(i);
        }

        Collections.shuffle(list);
        for(int i = 0; i < n; i++) {
            swapRows(a, i, list.get(i));
            swapRows(b, i, list.get(i));
        }
    }
}
