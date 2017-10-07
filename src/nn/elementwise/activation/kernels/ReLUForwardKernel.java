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
public class ReLUForwardKernel extends Kernel {

    private float[] input = new float[0];
    private float[] output = new float[0];
    
    public void call(float[] input, float[] output) {
        this.input = input;
        this.output = output;
        Range range = Range.create(input.length);
        execute(range);
    }
    
    @Override
    public void run() {
        int i = getGlobalId();
        
        output[i] = max(0, input[i]);
    }
    
}
