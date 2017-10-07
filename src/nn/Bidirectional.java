/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn;

/**
 *
 * @author bowen
 */
public interface Bidirectional {
    
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
    
}
