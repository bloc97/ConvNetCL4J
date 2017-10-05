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
public interface Layer {
    
    public float[] forward(float[] input);
    public float[] backward(float[] error);
    
    /**
     * @return Reference to the weights array of the layer.
     */
    public float[] getWeights();
    
    public int dimensions();
    public int[] getDimensionSizes();
    
}
