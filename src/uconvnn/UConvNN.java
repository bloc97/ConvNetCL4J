/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uconvnn;

import loss.MeanSquaredErrorLossFunction;
import loss.LossFunction;
import uconvnn.convolution.spatial.SpatialConvolutionLayer;
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
        
        int layerSize = 3;
        
        SpatialConvolutionLayer layer = new SpatialConvolutionLayer(layerSize, layerSize, 1, 1, 1, 1, 1, 1);
        
        
        float[] weights = layer.getWeights();
        
        for (int i=0; i<weights.length; i++) {
            weights[i] = (float)(Math.random() * 2d - 1);
        }
        
        float[] input1 = new float[] {0.2f,0.5f,0.1f,0.4f,0.9f,0.5f,0.7f,0.6f,0.0f};
        float[] expectedOutput1 = new float[] {0.9f,0.5f,0.0f,0.4f,0.1f,0.7f,0.7f,0.6f,0.5f};
        float[] input2 = new float[] {0.2f,-0.5f,0.5f,0.3f,0.2f,-0.4f,0.5f,0.6f,0.0f};
        float[] expectedOutput2 = new float[] {-0.5f,0.52f,0.2f,0.4f,0.6f,0.5f,0.1f,0.3f,0.4f};
        float[] input3 = new float[] {-0.5f,0.1f,0.3f,0.2f,0.7f,-0.2f,-0.4f,0.4f,0.1f};
        float[] expectedOutput3 = new float[] {0.2f,0.0f,0.4f,0.9f,0.7f,0.3f,-0.5f,-0.6f,0.0f};
        
        
        layer.setInputSize(3, 3, 1);
        
        float[] output = layer.forward(input1);
        float[] error = new float[output.length];
        
        for (int i=0; i<output.length; i++) {
            float diff = expectedOutput1[i] - output[i];
            error[i] = diff;
        }
        
        LossFunction lossFunction = new MeanSquaredErrorLossFunction();
        
        System.out.println(Arrays.toString(input1));
        System.out.println(Arrays.toString(expectedOutput1));
        System.out.println(Arrays.toString(output));
        System.out.println(Arrays.toString(error));
        
        float[] inputError = layer.backward(error);
        
        for (int i=0; i<100000; i++) {
            output = layer.forward(input1);
            error = lossFunction.getErrorDerivativeArray(output, expectedOutput1);
            inputError = layer.backward(error);
            layer.grad();
            System.out.println(meanSquaredError(output, expectedOutput1));
            
            output = layer.forward(input2);
            error = lossFunction.getErrorDerivativeArray(output, expectedOutput2);
            inputError = layer.backward(error);
            layer.grad();
            System.out.println(meanSquaredError(output, expectedOutput2));
            
            output = layer.forward(input3);
            error = lossFunction.getErrorDerivativeArray(output, expectedOutput3);
            inputError = layer.backward(error);
            layer.grad();
            System.out.println(meanSquaredError(output, expectedOutput3));
            
            float[] grad = layer.getGradients();
            
            for (int g=0; g<grad.length; g++) {
                weights[g] += grad[g] / 3 * 0.05f;
            }
            layer.resetGradients();
            //System.out.println(Arrays.toString(inputError));
            //System.out.println(Arrays.toString(grad));
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
