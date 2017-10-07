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
public interface Gradable {
    
    /**
     * Computes the gradient with respect to the last backward pass <br>
     * Note, this function does <br> oldGrad = oldGrad + newGrad <br><br>
     * To clear the gradient, use {@link resetGradients()}
     */
    public void grad();
    /**
     * Resets the gradient array to 0;
     */
    public void resetGradients();
    
}
