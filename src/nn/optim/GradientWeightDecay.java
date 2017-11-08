/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.optim;

import java.util.HashSet;
import java.util.Set;
import nn.Layer;
import nn.Network;
import nn.NeuronLayer;
import nn.optim.kernels.GradientWeightDecayKernel;

/**
 *
 * @author bowen
 */
public class GradientWeightDecay {
    
    public final static GradientWeightDecayKernel KERNEL = new GradientWeightDecayKernel();
    
    public void decay(Network network, float weightDecay) {
        Set<Layer> visitedLayers = new HashSet<>();
        
        decay(network, weightDecay, visitedLayers);
    }
    
    private void decay(Network network, float weightDecay, Set<Layer> visitedLayers) {
        for (Layer layer : network.getLayers()) {
            if (layer instanceof Network) {
                decay((Network) layer, weightDecay, visitedLayers);
            } else if (layer instanceof NeuronLayer) {
                NeuronLayer nlayer = (NeuronLayer) layer;
                
                if (!nlayer.isGradientZero() && !visitedLayers.contains(layer)) {
                    KERNEL.call(nlayer.getWeights(), nlayer.getGradients(), weightDecay);
                    visitedLayers.add(layer);
                    System.out.print("Decay " + weightDecay + " | ");
                }
                
            }
        }
    }
}
