package com.deep4j;

import org.ejml.simple.SimpleMatrix;

public class TrainInfo {
    public SimpleMatrix obsTrain;
    public SimpleMatrix obsTest;
    public SimpleMatrix targetTrain;
    public SimpleMatrix targetTest;
    public int epochs = 100;
    public int evalEvery = 10;
    public int batchSize = 60;
    public int seed = 1;
    public boolean restart = true;
    public boolean earlyStopping = true;

    public TrainInfo(SimpleMatrix obsTrain,
                     SimpleMatrix targetTrain,
                     SimpleMatrix obsTest,
                     SimpleMatrix targetTest) {
        this.obsTrain = obsTrain;
        this.obsTest = obsTest;
        this.targetTrain = targetTrain;
        this.targetTest = targetTest;
    }
}