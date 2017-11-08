/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.elementwise.activation.kernels;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 *
 * @author bowen
 */
public class LeakyReLUForwardKernel extends Kernel {

    private float[] input = new float[0];
    private float[] output = new float[0];
    
    private float[] prop = new float[1];
    
    public void call(float[] input, float[] output, float leakiness) {
        this.input = input;
        this.output = output;
        this.prop[0] = leakiness;
        
        Range range = Range.create(input.length);
        execute(range);
    }
    
    @Override
    public void run() {
        int i = getGlobalId();
        
        if (input[i] > 0) {
            output[i] = input[i];
        } else {
            output[i] = prop[0] * input[i];
        }
    }
    
}
