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

/**
 *
 * @author bowen
 */
public class GradientClip {
    
    public final static GradientClipKernel KERNEL = new GradientClipKernel();
    
    public void clip(Network network, float clip) {
        Set<Layer> visitedLayers = new HashSet<>();
        
        clip(network, clip, visitedLayers);
    }
    
    private void clip(Network network, float clip, Set<Layer> visitedLayers) {
        for (Layer layer : network.getLayers()) {
            if (layer instanceof Network) {
                clip((Network) layer, clip, visitedLayers);
            } else if (layer instanceof NeuronLayer) {
                NeuronLayer nlayer = (NeuronLayer) layer;
                
                if (!nlayer.isGradientZero() && !visitedLayers.contains(layer)) {
                    KERNEL.call(nlayer.getGradients(), clip);
                    visitedLayers.add(layer);
                }
                
            }
        }
    }
}
