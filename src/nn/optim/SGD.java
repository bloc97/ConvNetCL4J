/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.optim;


import nn.Layer;
import nn.Network;
import nn.NeuronLayer;
import nn.optim.kernels.SGDKernel;

/**
 *
 * @author bowen
 */
public class SGD {
    
    public final static SGDKernel KERNEL = new SGDKernel();
    
    public void update(Network network, int batchSize, float learningRate) {
        update(network, batchSize, learningRate, Float.MAX_VALUE);
    }
    public void update(Network network, int batchSize, float learningRate, float clip) {
        for (Layer layer : network.getLayers()) {
            if (layer instanceof Network) {
                update((Network) layer, batchSize, learningRate, clip);
            } else if (layer instanceof NeuronLayer) {
                NeuronLayer nlayer = (NeuronLayer) layer;
                KERNEL.call(nlayer.getWeights(), nlayer.getGradients(), batchSize, learningRate, clip);
                nlayer.resetGradients();
            }
        }
    }
    
}
