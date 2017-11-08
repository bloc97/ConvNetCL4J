/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.optim.kernels;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 *
 * @author bowen
 */
public class SGDMomentumKernel extends Kernel {
    
    private float[] weights = new float[0];
    private float[] velocity = new float[0];
    private float[] gradients = new float[0];
    
    private final float[] prop = new float[6];
    
    public void call(float[] weights, float[] velocity, float[] gradients, int batchSize, float learningRate, float momentum, float weightDecay, float clip, float velClip) {
        this.weights = weights;
        this.velocity = velocity;
        this.gradients = gradients;
        prop[0] = learningRate;
        prop[1] = clip;
        prop[2] = batchSize;
        prop[3] = momentum;
        prop[4] = weightDecay;
        prop[5] = velClip;
        
        Range range = Range.create(weights.length);
        execute(range);
    }
    
    @Override
    public void run() {
        int i = getGlobalId();
        
        float effectiveGradient = min(max((gradients[i] / prop[2]), -prop[1]), prop[1]); //Clipping
        
        float effectiveCost = effectiveGradient - (prop[4] * weights[i]); //l2 Weight decay
        
        velocity[i] = (prop[3] * velocity[i]) + (prop[0] * effectiveCost);
        //Add velocity clipping, keep momentum but update gets clipped
        
        float effectiveVelocity = min(max(velocity[i], -prop[5]), prop[5]);
        
        weights[i] = weights[i] + effectiveVelocity;
    }
    
}
