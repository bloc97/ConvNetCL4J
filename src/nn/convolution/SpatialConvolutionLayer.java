/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.convolution;

import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import nn.NeuronLayer;
import nn.convolution.kernels.BackwardSpatialConvolutionKernel;
import nn.convolution.kernels.ForwardSpatialConvolutionKernel;
import nn.convolution.kernels.GradSpatialConvolutionKernel;

/**
 *
 * @author bowen
 */
public class SpatialConvolutionLayer implements NeuronLayer {
    
    public final static ForwardSpatialConvolutionKernel FORWARDKERNEL = new ForwardSpatialConvolutionKernel();
    public final static BackwardSpatialConvolutionKernel BACKWARDKERNEL = new BackwardSpatialConvolutionKernel();
    public final static GradSpatialConvolutionKernel GRADKERNEL = new GradSpatialConvolutionKernel();
    
    private float[] weights = new float[0];
    private float[] gradients = new float[0];
    private final int[] kernelSize = new int[4]; //Width, Height, Depth, Number
    private final int[] kernelDim = new int[4]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private final int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private final int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] input = new float[0];
    private float[] inputError = new float[0];
    private final int[] inputSize = new int[4]; //Width, Height, Depth, N
    private final int[] inputDim = new int[4]; //Width, Width * Height, Length, Total Length
    
    private float[] output = new float[0];
    private float[] outputError = new float[0];
    private final int[] outputSize = new int[4];
    private final int[] outputDim = new int[4];
    
    private boolean isGradientEmpty = true;
    
    public SpatialConvolutionLayer(int w, int h , int d, int n, int shorz, int svert, int phorz, int pvert) { //TODO: Allow non-odd kernels and individual side padding

        int length = w * h * d * n + n;
        this.weights = new float[length];
        gradients = new float[length];
        kernelSize[0] = w;
        kernelSize[1] = h;
        kernelSize[2] = d;
        kernelSize[3] = n;
        
        kernelDim[0] = w;
        kernelDim[1] = w * h;
        kernelDim[2] = w * h * d + 1; //1 array entry for bias
        kernelDim[3] = w * h * d * n + n; //n array entries for biases
        
        stride[0] = shorz;
        stride[1] = svert;
        padding[0] = phorz;
        padding[1] = pvert;
    }
    
    private SpatialConvolutionLayer(float[] weights, int[] kernelSize, int[] stride, int[] padding) {
        this(kernelSize[0], kernelSize[1], kernelSize[2], kernelSize[3], stride[0], stride[1], padding[0], padding[1]);
        this.weights = weights;
    }
    
    public NeuronLayer createSharedClone() {
        return new SpatialConvolutionLayer(weights, kernelSize, stride, padding);
    }
    
    @Override
    public void setInputSize(int[] size) {
        if (size.length != 4) {
            throw new IllegalArgumentException("Wrong input dimension.");
        }
        int w = size[0];
        int h = size[1];
        int d = size[2];
        int n = size[3];
        
        if (d != kernelSize[2]) {
            throw new IllegalArgumentException("Wrong input depth.");
        }
        int length = w * h * d * n;
        input = new float[length];
        inputError = new float[length];
        inputSize[0] = w;
        inputSize[1] = h;
        inputSize[2] = d;
        inputSize[3] = n;
        
        inputDim[0] = w;
        inputDim[1] = w * h;
        inputDim[2] = w * h * d;
        inputDim[3] = w * h * d * n;
        
        outputSize[0] = (inputSize[0] - kernelSize[0] + (2 * padding[0])) / stride[0] + 1;
        outputSize[1] = (inputSize[1] - kernelSize[1] + (2 * padding[1])) / stride[1] + 1;
        outputSize[2] = kernelSize[3];
        outputSize[3] = n;
        
        outputDim[0] = outputSize[0];
        outputDim[1] = outputSize[0] * outputSize[1];
        outputDim[2] = outputSize[0] * outputSize[1] * outputSize[2];
        outputDim[3] = outputSize[0] * outputSize[1] * outputSize[2] * outputSize[3];
        
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
    public int[] getWeightSize() {
        return kernelSize;
    }
    
    public float[] getInputError() {
        return inputError;
    }
    public float[] getOutput() {
        return output;
    }
    @Override
    public float[] getWeights() {
        return weights;
    }
    @Override
    public float[] getGradients() {
        return gradients;
    }
    
    @Override
    public void resetGradients() {
        Arrays.fill(gradients, 0);
        isGradientEmpty = true;
    }

    @Override
    public boolean isGradientEmpty() {
        return isGradientEmpty;
    }

    @Override
    public float[] forward(float[] input) {
        if (input.length != this.input.length) {
            throw new IllegalArgumentException("Wrong input array size.");
        }
        this.input = input;
        
        FORWARDKERNEL.call(weights, kernelSize, kernelDim, stride, padding, input, inputSize, inputDim, output, outputSize, outputDim);
        
        return output;
    }

    @Override
    public float[] backward(float[] outputError) {
        if (outputError.length != output.length) {
            throw new IllegalArgumentException("Wrong output error array size.");
        }
        this.outputError = outputError;
        
        BACKWARDKERNEL.call(weights, kernelSize, kernelDim, stride, padding, outputError, outputSize, outputDim, inputError, inputSize, inputDim);
        
        return inputError;
    }

    @Override
    public void grad() {
        GRADKERNEL.call(weights, gradients, kernelSize, kernelDim, stride, padding, input, inputSize, inputDim, outputError, outputSize, outputDim);
        isGradientEmpty = false;
    }

    @Override
    public int getFanIn() {
        return kernelSize[0] * kernelSize[1] * kernelSize[2];
    }

    @Override
    public int getFanOut() {
        return kernelSize[0] * kernelSize[1] * kernelSize[3];
    }

    
}
