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
public class LeakyReLUBackwardKernel extends Kernel {

    private float[] input = new float[0];
    private float[] outputError = new float[0];
    private float[] inputError = new float[0];
    
    private float[] prop = new float[1];

    public void call(float[] input, float[] outputError, float[] inputError, float leakiness) {
        this.input = input;
        this.outputError = outputError;
        this.inputError = inputError;
        this.prop[0] = leakiness;
        
        Range range = Range.create(outputError.length);
        execute(range);
    }
    
    @Override
    public void run() {
        int i = getGlobalId();
        
        if (input[i] < 0) {
            inputError[i] = prop[0] * outputError[i];
        } else {
            inputError[i] = outputError[i];
        }
    }
    
}
