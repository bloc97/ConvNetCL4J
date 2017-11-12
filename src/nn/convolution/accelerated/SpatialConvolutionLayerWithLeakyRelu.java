/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.convolution.accelerated;

import nn.convolution.*;

/**
 *
 * @author bowen
 */
public class SpatialConvolutionLayerWithLeakyRelu extends SpatialConvolutionLayer {
    
    public final static ForwardSpatialConvolutionWithLeakyReluKernel FORWARDKERNELWITHRELU = new ForwardSpatialConvolutionWithLeakyReluKernel();
    public final static BackwardSpatialConvolutionWithLeakyReluKernel BACKWARDKERNELWITHRELU = new BackwardSpatialConvolutionWithLeakyReluKernel();

    private float a = 0;
    
    public SpatialConvolutionLayerWithLeakyRelu(int w, int h, int d, int n, int shorz, int svert, int phorz, int pvert, float leakiness) {
        super(w, h, d, n, shorz, svert, phorz, pvert);
        this.a = leakiness;
    }
    
    
    @Override
    public float[] forward(float[] input) {
        if (input.length != this.input.length) {
            throw new IllegalArgumentException("Wrong input array size.");
        }
        this.input = input;
        
        FORWARDKERNELWITHRELU.call(weights, kernelSize, kernelDim, stride, padding, input, inputSize, inputDim, output, outputSize, outputDim, a);
        
        return output;
    }

    @Override
    public float[] backward(float[] outputError) {
        if (outputError.length != output.length) {
            throw new IllegalArgumentException("Wrong output error array size.");
        }
        this.outputError = outputError;
        
        BACKWARDKERNELWITHRELU.call(weights, kernelSize, kernelDim, stride, padding, outputError, outputSize, outputDim, inputError, inputSize, inputDim, a);
        
        return inputError;
    }

    
}
