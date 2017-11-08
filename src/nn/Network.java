/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

/**
 *
 * @author bowen
 */
public class Network implements Layer, Gradable {
    
    private final List<Layer> layers;

    public Network() {
        this.layers = new LinkedList<>();
    }
    
    /**
     * @return List of references to the layers of the network.
     */
    public List<Layer> getLayers() {
        return layers;
    }
    
    public void addLayer(Layer layer) {
        this.layers.add(layer);
    }
    
    @Override
    public float[] forward(float[] input) {
        float[] data = input;
        for (Layer layer : layers) {
            data = layer.forward(data);
        }
        return data;
    }

    @Override
    public float[] backward(float[] outputError) {
        float[] data = outputError;
        
        ListIterator<Layer> li = layers.listIterator(layers.size());
        while(li.hasPrevious()) {
            data = li.previous().backward(data);
        }
        return data;
    }
    
    @Override
    public void grad() {
        for (Layer layer : layers) {
            if (layer instanceof Gradable) {
                Gradable nlayer = (Gradable) layer;
                nlayer.grad();
            }
        }
    }

    @Override
    public void resetGradients() {
        for (Layer layer : layers) {
            if (layer instanceof Gradable) {
                Gradable nlayer = (Gradable) layer;
                nlayer.resetGradients();
            }
        }
    }

    @Override
    public boolean isGradientZero() {
        for (Layer layer : layers) {
            if (layer instanceof Gradable) {
                Gradable nlayer = (Gradable) layer;
                if (!nlayer.isGradientZero()) {
                    return false;
                }
            }
        }
        return true;
    }
    
    

    @Override
    public void setInputSize(int[] size) {
        int[] data = size;
        for (Layer layer : layers) {
            layer.setInputSize(data);
            data = layer.getOutputSize();
        }
    }

    @Override
    public int[] getInputSize() {
        if (layers.isEmpty()) {
            return new int[0];
        } else {
            return layers.get(0).getInputSize();
        }
    }

    @Override
    public int[] getOutputSize() {
        if (layers.isEmpty()) {
            return new int[0];
        } else {
            return layers.get(layers.size() - 1).getOutputSize();
        }
    }
}
