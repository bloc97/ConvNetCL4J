/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn;

import nn.loss.MeanSquaredErrorLossFunction;
import nn.loss.LossFunction;
import nn.convolution.SpatialConvolutionLayer;
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
        
        int x = 14;
        int y = 14;
        
        SpatialConvolutionLayer layer = new SpatialConvolutionLayer(layerSize, layerSize, 3, 3, 1, 1, 1, 1);
        
        
        float[] weights = layer.getWeights();
        
        for (int i=0; i<weights.length; i++) {
            weights[i] = (float)(Math.random() * 2d - 1);
        }
        /*
        float[] input1 = new float[] {0.2f,0.5f,0.1f,0.4f,0.9f,0.5f,0.7f,0.6f,0.0f};
        float[] expectedOutput1 = new float[] {0.9f,0.5f,0.0f,0.4f,0.1f,0.7f,0.7f,0.6f,0.5f};
        float[] input2 = new float[] {0.2f,-0.5f,0.5f,0.3f,0.2f,-0.4f,0.5f,0.6f,0.0f};
        float[] expectedOutput2 = new float[] {-0.5f,0.52f,0.2f,0.4f,0.6f,0.5f,0.1f,0.3f,0.4f};
        float[] input3 = new float[] {-0.5f,0.1f,0.3f,0.2f,0.7f,-0.2f,-0.4f,0.4f,0.1f};
        float[] expectedOutput3 = new float[] {0.2f,0.0f,0.4f,0.9f,0.7f,0.3f,-0.5f,-0.6f,0.0f};
        */
        
        float[] input1 = new float[x*y*3];
        float[] expectedOutput1 = new float[x*y*3];
        float[] input2 = new float[x*y*3];
        float[] expectedOutput2 = new float[x*y*3];
        float[] input3 = new float[x*y*3];
        float[] expectedOutput3 = new float[x*y*3];
        
        for (int i=0; i<input1.length; i++) {
            input1[i] = (float)(Math.random() * 2d - 1);
            input2[i] = (float)(Math.random() * 2d - 1);
            input3[i] = (float)(Math.random() * 2d - 1);
            expectedOutput1[i] = (float)(Math.random() * 2d - 1);
            expectedOutput2[i] = (float)(Math.random() * 2d - 1);
            expectedOutput3[i] = (float)(Math.random() * 2d - 1);

        }
        
        layer.setInputSize(new int[] {x, y, 3});
        
        float[] output = layer.forward(input1);
        float[] error = new float[output.length];
        
        for (int i=0; i<output.length; i++) {
            float diff = expectedOutput1[i] - output[i];
            error[i] = diff;
        }
        
        LossFunction lossFunction = new MeanSquaredErrorLossFunction();
        
        //System.out.println(Arrays.toString(input1));
        //System.out.println(Arrays.toString(expectedOutput1));
        //System.out.println(Arrays.toString(output));
        //System.out.println(Arrays.toString(error));
        
        float[] inputError = layer.backward(error);
        
        for (int i=0; i<100000; i++) {
            
            float totalError = 0;
            
            output = layer.forward(input1);
            error = lossFunction.getErrorDerivativeArray(output, expectedOutput1);
            inputError = layer.backward(error);
            layer.grad();
            totalError += lossFunction.getError(output, expectedOutput1);
            
            output = layer.forward(input2);
            error = lossFunction.getErrorDerivativeArray(output, expectedOutput2);
            inputError = layer.backward(error);
            layer.grad();
            totalError += lossFunction.getError(output, expectedOutput2);
            
            output = layer.forward(input3);
            error = lossFunction.getErrorDerivativeArray(output, expectedOutput3);
            inputError = layer.backward(error);
            layer.grad();
            totalError += lossFunction.getError(output, expectedOutput3);
            
            totalError /= 3;
            
            System.out.println("i: " + i + " " + totalError);
            
            float[] grad = layer.getGradients();
            
            for (int g=0; g<grad.length; g++) {
                weights[g] += grad[g] / 3 * 0.05f;
            }
            layer.resetGradients();
            //System.out.println(Arrays.toString(inputError));
            //System.out.println(Arrays.toString(grad));
            //System.out.println(Arrays.toString(error));
        }
        
        
    }
    
    
    
}
