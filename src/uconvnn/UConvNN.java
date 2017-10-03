/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uconvnn;

import com.aparapi.Kernel;
import com.aparapi.Range;
import java.util.Arrays;

/**
 *
 * @author bowen
 */
public class UConvNN {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        ForwardSpatialConvolutionKernel kernel = new ForwardSpatialConvolutionKernel();
        
        float[] layer = new float[] {0, -1, 0, 1, -1, 0, 0, 1, 0,0,-1,0,1,-1,1,1,0,-1,1,1,-1,0,1,0,-1,-1,1,1    , 0,-1,1,1,-1,0,0,-1,0,0,-1,1,1,0,-1,-1,1,0,0,0,-1,-1,-1,1,0,-1,-1,0};
        //float[] input = new float[] {2,2,2,0,2,1,0,2,1,2,2,1,0,2,1,0,2,2,0,1,2,0,0,1,0,2,2,1,1,2,2,0,1,2,1,1,0,0,0,1,2,1,0,2,2,1,0,0,1,2,1,0,0,2,1,0,0,1,0,2,0,0,1,2,0,0,0,0,2,1,0,0,1,1,0};
        float[] input = new float[10000*40000*3];
        kernel.setLayer(layer, 3, 3, 3, 2, 2, 2, 1, 1);
        kernel.setInput(input, 10000, 40000, 3);
        
        //System.out.println(kernel.getFromLayer(1,0,0,1));
        while(true) {
            long s = System.currentTimeMillis();
            kernel.forward();
            //System.out.println(Arrays.toString(kernel.getOutput()));
            System.out.println(System.currentTimeMillis() - s);
        }
        
    }
    
}
