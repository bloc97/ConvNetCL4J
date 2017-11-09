/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn;

import java.util.Random;

/**
 *
 * @author bowen
 */
public abstract class Randomiser {
    
    public static void uniform(NeuronLayer layer, float mean, float multiplier, Random random) {
        float[] weights = layer.getWeights();
        
        float min = -(multiplier) + mean;
        
        for (int i=0; i<weights.length; i++) {
            weights[i] = (random.nextFloat() * 2 * multiplier + min); 
        }
    }
    public static void uniform(NeuronLayer layer, float mean, float multiplier) {
        Random random = new Random();
        uniform(layer, mean, multiplier, random);
    }
    
    public static void normal(NeuronLayer layer, float mean, float multiplier, Random random) {
        float[] weights = layer.getWeights();
        
        for (int i=0; i<weights.length; i++) {
            weights[i] = (float)(random.nextGaussian() * multiplier + mean); 
        }
    }
    public static void normal(NeuronLayer layer, float mean, float multiplier) {
        Random random = new Random();
        normal(layer, mean, multiplier, random);
    }
    
    public static void kaimingUniform(NeuronLayer layer) {
        Random random = new Random();
        kaimingUniform(layer, random);
    }
    public static void kaimingUniform(NeuronLayer layer, Random random) {
        uniform(layer, 0, (float)Math.sqrt(6f/layer.getFanIn()), random);
    }
    
}
