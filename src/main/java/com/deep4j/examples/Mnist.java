package com.deep4j.examples;

import com.deep4j.Network;
import com.deep4j.TrainInfo;
import com.deep4j.Trainer;
import com.deep4j.activations.Linear;
import com.deep4j.activations.Tanh;
import com.deep4j.layers.Dense;
import com.deep4j.losses.SoftmaxCrossEntropy;
import com.deep4j.optimizers.Optimizer;
import com.deep4j.optimizers.SGD;
import com.deep4j.optimizers.SGDMomentum;
import org.ejml.simple.SimpleMatrix;

import java.io.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.deep4j.utils.Matrix.elementApply;

public class Mnist {
    private static File getFile(String fileName) {
        URL url = Mnist.class.getClassLoader().getResource(fileName);
        if(url == null) {
            throw new RuntimeException("Unable to load file: " + fileName);
        }

        return new File(url.getFile());
    }

    private static SimpleMatrix readMnist(String fileName, int cols) {
        File file = getFile(fileName);
        List<double[]> rows = new ArrayList<>();
        try {
            DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
            dataInputStream.readInt();
            int numberOfItems = dataInputStream.readInt();
            for(int i = 0; i < numberOfItems; i++) {
                double[] data = new double[cols];
                for (int c = 0; c < cols; c++) {
                    data[c] = dataInputStream.readUnsignedByte();
                }
                rows.add(data);
            }
            dataInputStream.close();

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }

        SimpleMatrix mat = new SimpleMatrix(rows.size(), cols);
        for(int r = 0; r < rows.size(); r++) {
            mat.setRow(r, 0, rows.get(r));
        }

        return mat;
    }

    private static void normalize(TrainInfo tInfo) {
        double mean = tInfo.obsTrain.elementSum() / (double)tInfo.obsTrain.getNumElements();
        SimpleMatrix diff = elementApply(tInfo.obsTrain, (r, c, v) -> Math.pow(v - mean, 2));
        double std = Math.sqrt(diff.elementSum() / (double)tInfo.obsTrain.getNumElements());
        tInfo.obsTrain = elementApply(tInfo.obsTrain, (r, c, v) -> v - mean);
        tInfo.obsTest = elementApply(tInfo.obsTest, (r, c, v) -> v - mean);

        tInfo.obsTrain = tInfo.obsTrain.scale(1.0/ std);
        tInfo.obsTest = tInfo.obsTest.scale(1.0/ std);
    }

    // Do a one hot encoding
    private static SimpleMatrix encodeLabels(SimpleMatrix labels) {
        int c = 10;
        int r = labels.numRows();
        SimpleMatrix mat = new SimpleMatrix(r, c);
        return elementApply(mat, (row, col, v) -> {
           if(labels.get(row, 0) == col) {
               return 1;
           }

           return 0;
        });
    }

    private static int argMax(SimpleMatrix mat) {
        double[] data = mat.getDDRM().data;
        int maxIdx = 0;
        for(int i = 0; i < data.length; i++) {
            if(data[i] > data[maxIdx]) {
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    static void calcAccuracyModel(Network net, TrainInfo tInfo) {
        SimpleMatrix predictions = net.forward(tInfo.obsTest);
        int r = tInfo.targetTest.numRows();
        double sum = 0.0;
        for(int i =0; i < r; i++) {
            int pred = argMax(predictions.extractVector(true, i));
            int actual = argMax(tInfo.targetTest.extractVector(true, i));
            if(pred == actual) {
                sum++;
            }
        }

        System.out.println("\nThe model validation accuracy is : " +  sum / r);
    }

    private static void trainMnist(boolean momentum) {
        int seed = 20190119;
        TrainInfo trainInfo = new TrainInfo(
                readMnist("train-images.idx3-ubyte", 28 * 28),
                readMnist("train-labels.idx1-ubyte", 1),
                readMnist("t10k-images.idx3-ubyte", 28 * 28),
                readMnist("t10k-labels.idx1-ubyte", 1),
                Mnist::calcAccuracyModel
        );
        normalize(trainInfo);
        trainInfo.targetTrain = encodeLabels(trainInfo.targetTrain);
        trainInfo.targetTest = encodeLabels(trainInfo.targetTest);
        Network network = new Network(Arrays.asList(
                new Dense(89, seed, new Tanh()),
                new Dense(10, seed, new Linear())
        ), seed, new SoftmaxCrossEntropy(1e-9));
        Optimizer optimizer;
        if(momentum) {
            optimizer = new SGDMomentum(0.1, 0.9, network);
        } else {
            optimizer = new SGD(0.1, network);
        }

        Trainer trainer = new Trainer(optimizer);
        trainInfo.epochs = 50;
        trainInfo.evalEvery = 10;
        trainInfo.seed = seed;
        trainInfo.batchSize = 60;
        trainer.fit(trainInfo);
        System.out.println();

        calcAccuracyModel(network, trainInfo);
    }

    public static void main(String[] args) {
        trainMnist(true);
    }
}
