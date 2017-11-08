/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.optim;

import java.util.HashMap;
import java.util.Map;
import nn.Layer;
import nn.Network;
import nn.NeuronLayer;
import nn.optim.kernels.SGDMomentumKernel;

/**
 *
 * @author bowen
 */
public class SGDMomentum {
    
    public final static SGDMomentumKernel KERNEL = new SGDMomentumKernel();
    
    public final Map<NeuronLayer, float[]> velocityMap = new HashMap<>();
    
    public void update(Network network, int batchSize, float learningRate, float momentum) {
        update(network, batchSize, learningRate, momentum, 0, Float.MAX_VALUE);
    }
    public void update(Network network, int batchSize, float learningRate, float momentum, float weightDecay, float clip) {
        for (Layer layer : network.getLayers()) {
            if (layer instanceof Network) {
                update((Network) layer, batchSize, learningRate, momentum, weightDecay, clip);
            } else if (layer instanceof NeuronLayer) {
                NeuronLayer nlayer = (NeuronLayer) layer;
                
                if (!velocityMap.containsKey(nlayer)) {
                    velocityMap.put(nlayer, new float[nlayer.getWeights().length]);
                }
                
                KERNEL.call(nlayer.getWeights(), velocityMap.get(nlayer), nlayer.getGradients(), batchSize, learningRate, momentum, weightDecay, clip);
                nlayer.resetGradients();
            }
        }
    }
    
}
