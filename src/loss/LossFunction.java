/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package loss;

/**
 *
 * @author bowen
 */
public interface LossFunction {
    
    public float getError(float[] observed, float[] expected);
    public float getErrorDerivative(float observed, float expected);
    public float[] getErrorDerivativeArray(float[] observed, float[] expected);
    
    public float getBatchError(float[][] observedBatch, float[][] expectedBatch);
    
}
