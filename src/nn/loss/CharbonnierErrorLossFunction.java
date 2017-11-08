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
public class CharbonnierErrorLossFunction implements LossFunction {

    private final float epsilon, epsilonSqr;
    
    public CharbonnierErrorLossFunction(float epsilon) {
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
            totalError += Math.sqrt((diff * diff) + (epsilonSqr));
        }
        totalError /= observed.length;
        return totalError;
    }

    @Override
    public float getErrorDerivative(float observed, float expected) {
        float diff = expected - observed;
        
        return (float)(diff / (Math.sqrt(diff * diff + epsilonSqr)));
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
