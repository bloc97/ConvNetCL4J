/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.elementwise.activation;

import nn.Layer;

/**
 *
 * @author bowen
 */
public interface NonLinearActivationLayer extends Layer {
    
    public float equation(float x);
    public float derivative(float x);
    
}
