/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.convolution;

import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import nn.convolution.kernels.BackwardSpatialConvolutionKernel;
import nn.convolution.kernels.GradSpatialConvolutionKernel;
import nn.convolution.kernels.ForwardSpatialConvolutionKernel;
import nn.NeuronLayer;
import nn.convolution.kernels.BackwardSpatialTransposedConvolutionKernel;
import nn.convolution.kernels.ForwardSpatialTransposedConvolutionKernel;
import nn.convolution.kernels.GradSpatialTransposedConvolutionKernel;

/**
 *
 * @author bowen
 */
public class SpatialTransposedConvolutionLayer implements NeuronLayer {
    
    public final static ForwardSpatialTransposedConvolutionKernel FORWARDKERNEL = new ForwardSpatialTransposedConvolutionKernel();
    public final static BackwardSpatialTransposedConvolutionKernel BACKWARDKERNEL = new BackwardSpatialTransposedConvolutionKernel();
    public final static GradSpatialTransposedConvolutionKernel GRADKERNEL = new GradSpatialTransposedConvolutionKernel();
    
    private final LinkedList<Checkpoint> inputCheckpoints = new LinkedList<>();
    private final LinkedList<Checkpoint> errorCheckpoints = new LinkedList<>();
    
    private float[] weights = new float[0];
    private float[] gradients = new float[0];
    private final int[] kernelSize = new int[4]; //Width, Height, Depth, Number
    private final int[] kernelDim = new int[4]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private final int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private final int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] input = new float[0];
    private float[] inputError = new float[0];
    private final int inputSize[] = new int[3]; //Width, Height, Depth
    private final int inputDim[] = new int[3]; //Width, Width * Height, Total Length
    
    private float[] output = new float[0];
    private float[] outputError = new float[0];
    private final int[] outputSize = new int[3];
    private final int[] outputDim = new int[3]; //Width, Width * Height, Total Length
    
    private boolean isGradientZero = true;
    
    public static class Checkpoint {
        
        public final float[] data;
        
        public Checkpoint(float[] data) {
            this.data = Arrays.copyOf(data, data.length);
        }
    }
    
    
    public SpatialTransposedConvolutionLayer(int w, int h , int d, int n, int shorz, int svert, int phorz, int pvert) { //TODO: Allow non-odd kernels and individual side padding

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
    
    private SpatialTransposedConvolutionLayer(float[] weights, int[] kernelSize, int[] stride, int[] padding) {
        this(kernelSize[0], kernelSize[1], kernelSize[2], kernelSize[3], stride[0], stride[1], padding[0], padding[1]);
        this.weights = weights;
    }
    
    
    public NeuronLayer createSharedClone() {
        return new SpatialTransposedConvolutionLayer(weights, kernelSize, stride, padding);
    }
    
    @Override
    public void setInputSize(int[] size) {
        if (size.length != 3) {
            throw new IllegalArgumentException("Wrong input dimension.");
        }
        int w = size[0];
        int h = size[1];
        int d = size[2];
        
        if (d != kernelSize[3]) {
            throw new IllegalArgumentException("Wrong input depth.");
        }
        int length = w * h * d;
        input = new float[length];
        inputError = new float[length];
        inputSize[0] = w;
        inputSize[1] = h;
        inputSize[2] = d;
        
        inputDim[0] = w;
        inputDim[1] = w * h;
        inputDim[2] = w * h * d;
        
        
        outputSize[0] = (inputSize[0] - 1) * stride[0] + kernelSize[0] - (2 * padding[0]);
        outputSize[1] = (inputSize[1] - 1) * stride[1] + kernelSize[1] - (2 * padding[1]);
        outputSize[2] = kernelSize[2];

        outputDim[0] = outputSize[0];
        outputDim[1] = outputSize[0] * outputSize[1];
        outputDim[2] = outputSize[0] * outputSize[1] * outputSize[2];
        output = new float[outputDim[2]];
        outputError = new float[outputDim[2]];
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
        this.gradients = new float[kernelDim[3]];
        isGradientZero = true;
    }

    @Override
    public boolean isGradientZero() {
        return isGradientZero;
    }

    @Override
    public float[] forward(float[] input) {
        if (input.length != this.input.length) {
            throw new IllegalArgumentException("Wrong input array size.");
        }
        this.input = input;
        
        inputCheckpoints.add(new Checkpoint(input));
        FORWARDKERNEL.call(weights, kernelSize, kernelDim, stride, padding, input, inputSize, inputDim, output, outputSize, outputDim);
        
        return output;
    }

    @Override
    public float[] backward(float[] outputError) {
        if (outputError.length != output.length) {
            throw new IllegalArgumentException("Wrong output error array size.");
        }
        this.outputError = outputError;
        
        errorCheckpoints.addFirst(new Checkpoint(outputError));
        BACKWARDKERNEL.call(weights, kernelSize, kernelDim, stride, padding, outputError, outputSize, outputDim, inputError, inputSize, inputDim);
        
        return inputError;
    }

    @Override
    public void grad() {
        long startTime = System.currentTimeMillis();
        int size = inputCheckpoints.size();
        
        Iterator<Checkpoint> iit = inputCheckpoints.iterator();
        Iterator<Checkpoint> oet = errorCheckpoints.iterator();
        
        while(iit.hasNext() && oet.hasNext()) {
            float[] oetData = oet.next().data;
            float[] iitData = iit.next().data;
            GRADKERNEL.call(weights, gradients, kernelSize, kernelDim, stride, padding, oetData, outputSize, outputDim, iitData, inputSize, inputDim);
        }
        inputCheckpoints.clear();
        errorCheckpoints.clear();
        long endTime = System.currentTimeMillis();
        System.out.print("Grad " + size + " " + (endTime-startTime) + " ms | ");
        
        isGradientZero = false;
    }

    @Override
    public int getFanIn() {
        return kernelSize[0] * kernelSize[1] * kernelSize[3];
    }

    @Override
    public int getFanOut() {
        return kernelSize[0] * kernelSize[1] * kernelSize[2];
    }

    
}
