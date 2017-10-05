/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uconvnn;

import java.util.Random;

/**
 *
 * @author bowen
 */
public abstract class Randomiser {
    
    public static void uniform(Layer layer, float mean, float multiplier, Random random) {
        float[] weights = layer.getWeights();
        
        for (int i=0; i<weights.length; i++) {
            weights[i] = (random.nextFloat() * multiplier + mean); 
        }
    }
    public static void uniform(Layer layer, float mean, float multiplier) {
        Random random = new Random();
        uniform(layer, mean, multiplier, random);
    }
    
    public static void normal(Layer layer, float mean, float multiplier, Random random) {
        float[] weights = layer.getWeights();
        
        for (int i=0; i<weights.length; i++) {
            weights[i] = (float)(random.nextGaussian() * multiplier + mean); 
        }
    }
    public static void normal(Layer layer, float mean, float multiplier) {
        Random random = new Random();
        normal(layer, mean, multiplier, random);
    }
    
}
