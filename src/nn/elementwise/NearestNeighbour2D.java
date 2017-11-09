/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.elementwise;

import nn.Layer;
import nn.elementwise.kernels.BackwardNearestNeighbour2DKernel;
import nn.elementwise.kernels.ForwardNearestNeighbour2DKernel;

/**
 *
 * @author bowen
 */
public class NearestNeighbour2D implements Layer {
    public final static ForwardNearestNeighbour2DKernel FORWARDKERNEL = new ForwardNearestNeighbour2DKernel();
    public final static BackwardNearestNeighbour2DKernel BACKWARDKERNEL = new BackwardNearestNeighbour2DKernel();
    
    private float[] input = new float[0];
    private float[] inputError = new float[0];
    private final int[] inputSize = new int[4]; //Width, Height, Depth, N
    private final int[] inputDim = new int[4]; //Width, Width * Height, Length, Total Length
    
    private float[] output = new float[0];
    private float[] outputError = new float[0];
    private final int[] outputSize = new int[4];
    private final int[] outputDim = new int[4]; //Width, Width * Height, Length, Total Length

    @Override
    public void setInputSize(int[] size) {
        if (size.length != 4) {
           throw new IllegalArgumentException("Input dimension must be 4");
        }
        
        for (int i=0; i<size.length; i++) {
            inputSize[i] = size[i];
        }
        
        outputSize[0] = inputSize[0] * 2;
        outputSize[1] = inputSize[1] * 2;
        outputSize[2] = inputSize[2];
        outputSize[3] = inputSize[3];
        
        inputDim[0] = inputSize[0];
        inputDim[1] = inputSize[0] * inputSize[1];
        inputDim[2] = inputSize[0] * inputSize[1] * inputSize[2];
        inputDim[3] = inputSize[0] * inputSize[1] * inputSize[2] * inputSize[3];
        
        outputDim[0] = outputSize[0];
        outputDim[1] = outputSize[0] * outputSize[1];
        outputDim[2] = outputSize[0] * outputSize[1] * outputSize[2];
        outputDim[3] = outputSize[0] * outputSize[1] * outputSize[2] * outputSize[3];
        
        input = new float[inputDim[3]];
        inputError = new float[inputDim[3]];
        output = new float[outputDim[3]];
        outputError = new float[outputDim[3]];
        
    }

    @Override
    public int[] getInputSize() {
        return inputSize;
    }

    @Override
    public int[] getOutputSize() {
        return outputSize;
    }

    @Override
    public float[] forward(float[] input) {
        if (input.length != this.input.length) {
            throw new IllegalArgumentException("Wrong input array size.");
        }
        this.input = input;
        FORWARDKERNEL.call(input, inputSize, inputDim, output, outputSize, outputDim);
        return output;
    }

    @Override
    public float[] backward(float[] outputError) {
        if (outputError.length != this.outputError.length) {
            throw new IllegalArgumentException("Wrong output error array size.");
        }
        this.outputError = outputError;
        BACKWARDKERNEL.call(outputError, outputSize, outputDim, inputError, inputSize, inputDim);
        return inputError;
    }
    
}
