/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.elementwise.activation;

import nn.elementwise.activation.kernels.ReLUBackwardKernel;
import nn.elementwise.activation.kernels.ReLUForwardKernel;

/**
 *
 * @author bowen
 */
public class ReLULayer implements NonLinearActivationLayer {
    
    public final static ReLUForwardKernel FORWARDKERNEL = new ReLUForwardKernel();
    public final static ReLUBackwardKernel BACKWARDKERNEL = new ReLUBackwardKernel();
    
    private float[] input = new float[0];
    private float[] inputError = new float[0];
    private int inputSize[] = new int[0];
    
    private float[] output = new float[0];
    private float[] outputError = new float[0];
    
    @Override
    public float equation(float x) {
        return Math.max(0, x);
    }

    @Override
    public float derivative(float x) {
        return (x < 0) ? 0 : 1; 
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
        if (input.length != this.input.length) {
            throw new IllegalArgumentException("Wrong input array size.");
        }
        this.input = input;
        FORWARDKERNEL.call(input, output);
        
        return output;
    }

    @Override
    public float[] backward(float[] outputError) {
        if (input.length != this.input.length) {
            throw new IllegalArgumentException("Wrong output error array size.");
        }
        this.outputError = outputError;
        BACKWARDKERNEL.call(input, outputError, inputError);
        
        return inputError;
    }
    
}
