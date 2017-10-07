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
    
    /**
     * Forward pass
     * @param input Input data
     * @return Output data
     */
    public float[] forward(float[] input);
    
    /**
     * Back-propagation pass
     * @param outputError Output error
     * @return Input error
     */
    public float[] backward(float[] outputError);
    
    /**
     * Computes the gradient with respect to the last backward pass <br>
     * Note, this function does <br> oldGrad = oldGrad + newGrad <br><br>
     * To clear the gradient, use {@link resetGradients()}
     * @return Gradient array
     */
    public float[] grad();
    
    /**
     * @return Reference to the weight array of the layer.
     */
    public float[] getWeights();
    
    /**
     * @return Reference to the weight gradient array of the layer.
     */
    public float[] getGradients();
    
    /**
     * @return Weight size of the layer as a n-dimensional vector in array form.
     */
    public int[] getWeightSize();
    
    /**
     * Resets the gradient array to 0;
     */
    public void resetGradients();
    
}
