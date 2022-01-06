package com.deep4j;

import com.deep4j.layers.Layer;
import com.deep4j.optimizers.Optimizer;
import org.ejml.simple.SimpleMatrix;

import java.util.*;

import static com.deep4j.utils.Matrix.permute;

public class Trainer {
    private Network net;
    private final Optimizer optim;
    private double bestLoss = 1e9;

    public Trainer(Optimizer optim) {
        this.net = optim.getNet();
        this.optim = optim;
    }

    private List<SimpleMatrix[]> genBatch(SimpleMatrix obs, SimpleMatrix target, int batchSize) {
        if(obs.numRows() != target.numRows()) {
            throw new IllegalArgumentException(
                    "features and target must have the same number of rows"
            );
        }
        int n = obs.numRows();
        List<SimpleMatrix[]> batches = new ArrayList<>();
        for(int i = 0; i < n; i+=batchSize) {
            batches.add(
                    new SimpleMatrix[]{
                            obs.rows(i, Math.min(n, i + batchSize)),
                            target.rows(i, Math.min(n, i + batchSize))
                    }
            );
        }

        return batches;
    }

    public void fit(TrainInfo tInfo) {
        if(tInfo.restart) {
            for(Layer layer: net.layers) {
                layer.setFirst(true);
            }

            this.bestLoss = 1e9;
        }
        Network lastModel = null;
        for(int e = 0; e < tInfo.epochs; e++) {
            if((e + 1) % tInfo.evalEvery == 0) {
                lastModel = net.copy();
            }

            permute(tInfo.obsTrain, tInfo.targetTrain);
            List<SimpleMatrix[]> batches = genBatch(tInfo.obsTrain, tInfo.targetTrain, tInfo.batchSize);
            for(SimpleMatrix[] batch: batches) {
                net.trainBatch(batch[0], batch[1]);
                optim.step();
            }

            if((e + 1) % tInfo.evalEvery == 0) {
                SimpleMatrix testPredictions = this.net.forward(tInfo.obsTest);
                double loss = this.net.loss.forward(testPredictions, tInfo.targetTest);

                if(tInfo.earlyStopping) {
                    if (loss < bestLoss) {
                        System.out.printf("\nValidation loss after %d epochs is %.3f", e + 1, loss);
                        tInfo.accuracyCalculator.calcModelAccuracy(net, tInfo);
                        bestLoss = loss;
                    } else {
                        System.out.println();
                        System.out.printf("Loss increased after epoch %d, final loss was %.3f", e+ 1, bestLoss);
                        this.net = lastModel;
                        break;
                    }
                } else {
                    System.out.printf("\nValidation loss after %d epochs is %.3f", e + 1, loss);
                }
            }
        }
    }
}
