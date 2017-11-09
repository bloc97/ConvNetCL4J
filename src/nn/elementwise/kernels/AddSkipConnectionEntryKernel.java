/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.elementwise.kernels;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 *
 * @author bowen
 */
public class AddSkipConnectionEntryKernel extends Kernel {
    
    private float[] sum = new float[0];
    
    private float[] augend = new float[0];
    private float[] addend = new float[0];

    public void call(float[] sum, float[] augend, float[] addend) {
        this.sum = sum;
        this.augend = augend;
        this.addend = addend;
        
        Range range = Range.create(sum.length);
        execute(range);
    }
    
    @Override
    public void run() {
        int i = getGlobalId(0);
        sum[i] = augend[i] + addend[i];
        
    }
    
}
