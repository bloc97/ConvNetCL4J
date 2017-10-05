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
public interface CostFunction {
    
    public float getError(float[] observed, float[] expected);
    public float getErrorDerivativeRespectToOutput(float observed, float expected);
    public float[] getErrorArray(float[] observed, float[] expected);
    
    public float getBatchError(float[][] observedBatch, float[][] expectedBatch);
    
}
