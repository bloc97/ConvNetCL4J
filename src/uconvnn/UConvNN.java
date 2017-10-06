/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uconvnn;

import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
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
        
        
        System.out.println(OpenCLDevice.listDevices(Device.TYPE.GPU));
        System.out.println(OpenCLDevice.listDevices(Device.TYPE.CPU));
        
        
        SpatialConvolutionKernel kernel = new SpatialConvolutionKernel();
        
        int layerSize = 3;
        
        float[] weights = new float[10];
        
        for (int i=0; i<weights.length; i++) {
            weights[i] = (float)(Math.random() * 2d - 1);
        }
        
        float[] input = new float[] {0.2f,0.5f,0.1f,0.4f,0.9f,0.5f,0.7f,0.6f,0.0f};
        kernel.setWeight(weights, layerSize, layerSize, 1, 1, 1, 1, 1, 1);
        kernel.setInput(input, 3, 3, 1);
        
        float[] expectedOutput = new float[] {0.9f,0.5f,0.0f,0.4f,0.1f,0.7f,0.7f,0.6f,0.5f};
        float[] output = kernel.forward();
        float[] error = new float[output.length];
        
        for (int i=0; i<output.length; i++) {
            float diff = expectedOutput[i] - output[i];
            error[i] = diff;
        }
        
        System.out.println(Arrays.toString(input));
        System.out.println(Arrays.toString(expectedOutput));
        System.out.println(Arrays.toString(output));
        System.out.println(Arrays.toString(error));
        
        kernel.setOutputError(error);
        float[] inputError = kernel.backward();
        
        for (int i=0; i<100000; i++) {
            output = kernel.forward();
            error = new float[output.length];
            
            
            for (int e=0; e<output.length; e++) {
                float diff = expectedOutput[e] - output[e];
                error[e] = diff;
            }
            
            kernel.setOutputError(error);
            float[] grad = kernel.grad(0.05f);
            
            for (int g=0; g<grad.length; g++) {
                weights[g] = weights[g] + grad[g];
            }
            
            System.out.println(meanSquaredError(output, expectedOutput));
            //System.out.println(Arrays.toString(error));
        }
        
        
        /*
        //System.out.println(kernel.getFromLayer(1,0,0,1));
        while(true) {
            long s = System.currentTimeMillis();
            kernel.forward();
            //System.out.println(Arrays.toString(kernel.getOutput()));
            System.out.println(System.currentTimeMillis() - s);
        }*/
        
    }
    
    public static float meanSquaredError(float[] output, float[] expected) {
        
        if (output.length != expected.length) {
            throw new IllegalArgumentException("Different array lengths.");
        }
        
        float error = 0;
        
        for (int i=0; i<output.length; i++) {
            float diff = output[i] - expected[i];
            error += diff * diff;
        }
        error /= output.length;
        
        return error;
    }
    
    
}
