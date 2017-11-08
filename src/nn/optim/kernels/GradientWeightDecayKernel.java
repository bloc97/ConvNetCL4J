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
public class GradientWeightDecayKernel extends Kernel {
    
    private float[] weights = new float[0];
    private float[] gradients = new float[0];
    
    private final float[] prop = new float[1];
    
    public void call(float[] weights, float[] gradients, float weightDecay) {
        this.weights = weights;
        this.gradients = gradients;
        prop[0] = weightDecay;
        
        Range range = Range.create(weights.length);
        execute(range);
    }
    
    @Override
    public void run() {
        int i = getGlobalId();
        
        gradients[i] = gradients[i] - (prop[0] * weights[i]); //l2 Weight decay
    }
    
}
