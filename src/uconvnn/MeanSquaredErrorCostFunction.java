/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uconvnn;

/**
 *
 * @author bowen
 */
public class MeanSquaredErrorCostFunction implements CostFunction {

    @Override
    public float getError(float[] observed, float[] expected) {
        if (observed.length != expected.length || observed.length == 0) {
            throw new IllegalArgumentException("Wrong array size.");
        }
        float totalError = 0;
        for (int i=0; i<observed.length; i++) {
            float diff = expected[i] - observed[i];
            totalError += (diff * diff) / 2;
        }
        totalError /= observed.length;
        return totalError;
    }

    @Override
    public float getErrorDerivativeRespectToOutput(float observed, float expected) {
        return expected - observed;
    }

    @Override
    public float[] getErrorArray(float[] observed, float[] expected) {
        if (observed.length != expected.length || observed.length == 0) {
            throw new IllegalArgumentException("Wrong array size.");
        }
        float[] error = new float[observed.length];
        for (int i=0; i<observed.length; i++) {
            error[i] = getErrorDerivativeRespectToOutput(observed[i], expected[i]);
        }
        return error;
    }
    

    @Override
    public float getBatchError(float[][] observedBatch, float[][] expectedBatch) {
        if (observedBatch.length != expectedBatch.length || observedBatch.length == 0) {
            throw new IllegalArgumentException("Wrong array size.");
        }
        float totalError = 0;
        for (int n=0; n<observedBatch.length; n++) {
            totalError += getError(observedBatch[n], expectedBatch[n]);
        }
        totalError /= observedBatch.length;
        return totalError;
    }

    
}
