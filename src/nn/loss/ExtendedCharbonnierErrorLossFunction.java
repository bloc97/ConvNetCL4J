/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.loss;

/**
 *
 * @author bowen
 */
public class ExtendedCharbonnierErrorLossFunction implements LossFunction {

    private final float epsilon, epsilonSqr;
    
    public ExtendedCharbonnierErrorLossFunction(float epsilon) {
        this.epsilon = epsilon;
        this.epsilonSqr = epsilon * epsilon;
    }
    
    
    @Override
    public float getError(float[] observed, float[] expected) {
        if (observed.length != expected.length || observed.length == 0) {
            throw new IllegalArgumentException("Wrong array size.");
        }
        float totalError = 0;
        for (int i=0; i<observed.length; i++) {
            float diff = expected[i] - observed[i];
            float diffSqr = diff * diff;
            float xOe = diff/epsilon;
            float xOeSqr = xOe * xOe;
            totalError += Math.sqrt(xOeSqr + 1) - 1;
        }
        totalError /= observed.length;
        return totalError;
    }

    @Override
    public float getErrorDerivative(float observed, float expected) {
        float diff = expected - observed;
        float xOe = diff/epsilon;
        float xOeSqr = xOe * xOe;
        return (float)(diff / (epsilonSqr * Math.sqrt(xOeSqr + 1)));
    }

    @Override
    public float[] getErrorDerivativeArray(float[] observed, float[] expected) {
        if (observed.length != expected.length || observed.length == 0) {
            throw new IllegalArgumentException("Wrong array size.");
        }
        float[] error = new float[observed.length];
        for (int i=0; i<observed.length; i++) {
            error[i] = getErrorDerivative(observed[i], expected[i]);
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
