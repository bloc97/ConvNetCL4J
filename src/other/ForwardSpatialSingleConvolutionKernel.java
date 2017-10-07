/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package other;

import com.aparapi.Kernel;

/**
 *
 * @author bowen
 */
public class ForwardSpatialSingleConvolutionKernel extends Kernel { //Ignore kernel depth, apply one kernel to all channels, otherwise same as spatialconvolution
    
    private float[] layer;
    private int kernelWidth, kernelHeight;
    private int kernelNum;
    
    private int strideWidth, strideHeight;
    private int paddingHorizontal, paddingVertical; //Padding on one side
    
    
    
    private float[] input;
    private int inputWidth, inputHeight, inputDepth;
    
    private float[] output;
    private int outputWidth, outputHeight, outputDepth;
    
    public void setLayer(float[] layer, int w, int h, int n, int shorz, int svert, int phorz, int pvert) {
        if (layer.length != w * h * n + n) {
            throw new IllegalArgumentException("Wrong layer or specified size.");
        }
        this.layer = layer;
        kernelWidth = w;
        kernelHeight = h;
        kernelNum = n;
        
        strideWidth = shorz;
        strideHeight = svert;
        paddingHorizontal = phorz;
        paddingVertical = pvert;
        
    }
    
    public void setInput(float[] input, int w, int h, int d) {
        if (input.length != w * h * d) {
            throw new IllegalArgumentException("Wrong input or specified size.");
        }
        this.input = input;
        inputWidth = w;
        inputHeight = h;
        inputDepth = d;
        
        outputWidth = (inputWidth - kernelWidth + (2 * paddingHorizontal)) / strideWidth + 1;
        outputHeight = (inputHeight - kernelHeight + (2 * paddingVertical)) / strideHeight + 1;
        outputDepth = kernelNum;
        output = new float[outputWidth * outputHeight * outputDepth];
    }
    
    public float[] getOutput() {
        return output;
    }
    
    public void forward() {
        
    }
    
    @Override
    public void run() {
        
        
        
        
    }
    
}
