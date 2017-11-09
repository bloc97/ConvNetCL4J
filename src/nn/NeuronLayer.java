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
public interface NeuronLayer extends Layer, Gradable {
    
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
    
    public int getFanIn();
    public int getFanOut();
    
    public Layer createSharedClone();
    
    public boolean isGradientEmpty();
    
}
