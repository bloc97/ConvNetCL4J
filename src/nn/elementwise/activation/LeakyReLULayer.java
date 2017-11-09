/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.elementwise.activation;

import nn.Layer;
import nn.elementwise.activation.kernels.LeakyReLUForwardKernel;
import nn.elementwise.activation.kernels.LeakyReLUBackwardKernel;
/**
 *
 * @author bowen
 */
public class LeakyReLULayer implements NonLinearActivationLayer {
    
    public final static LeakyReLUForwardKernel FORWARDKERNEL = new LeakyReLUForwardKernel();
    public final static LeakyReLUBackwardKernel BACKWARDKERNEL = new LeakyReLUBackwardKernel();
    
    private float[] input = new float[0];
    private float[] inputError = new float[0];
    private int inputSize[] = new int[0];
    private int inputLength = 0;
    
    private float[] output = new float[0];
    private float[] outputError = new float[0];
    
    private float a = 0;
    
    public LeakyReLULayer(float leakiness) {
        a = leakiness;
    }
    
    @Override
    public float equation(float x) {
        if (x > 0) {
            return x;
        } else {
            return a * x;
        }
    }

    @Override
    public float derivative(float x) {
        if (x > 0) {
            return 1;
        } else {
            return a;
        }
    }

    @Override
    public void setInputSize(int[] size) {
        if (size.length < 1) {
            throw new IllegalArgumentException("Illegal input dimension: " + size.length);
        }
        inputSize = size;
        int length = size[0];
        for (int i=1; i<inputSize.length; i++) {
            length *= size[i];
        }
        input = new float[length];
        inputError = new float[length];
        
        output = new float[length];
        outputError = new float[length];
        inputLength = length;
    }

    @Override
    public int[] getInputSize() {
        return inputSize;
    }

    @Override
    public int[] getOutputSize() {
        return inputSize;
    }

    @Override
    public float[] forward(float[] input) {
        if (input.length != inputLength) {
            throw new IllegalArgumentException("Wrong input array size.");
        }
        this.input = input;
        FORWARDKERNEL.call(input, output, a);
        
        return output;
    }

    @Override
    public float[] backward(float[] outputError) {
        if (outputError.length != inputLength) {
            throw new IllegalArgumentException("Wrong output error array size.");
        }
        this.outputError = outputError;
        BACKWARDKERNEL.call(input, outputError, inputError, a);
        
        return inputError;
    }
    
}
