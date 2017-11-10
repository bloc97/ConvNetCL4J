/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.convolution.kernels;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;

/**
 *
 * @author bowen
 */
public class ForwardSpatialConvolutionKernel extends Kernel { //Sums the result of kernel depths, normal convolution
    
    private float[] weights = new float[0];
    private int[] kernelSize = new int[4]; //Width, Height, Depth, Number
    private int[] kernelDim = new int[4]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] input = new float[0];
    private int[] inputSize = new int[4]; //Width, Height, Depth, N
    private int[] inputDim = new int[4]; //Width, Width * Height, Length, Total Length
    
    private float[] output = new float[0];
    private int[] outputSize = new int[4];
    private int[] outputDim = new int[4];

    public void call(float[] weights, int[] kernelSize, int[] kernelDim, int[] stride, int padding[], float[] input, int[] inputSize, int[] inputDim, float[] output, int[] outputSize, int[] outputDim) {
        this.weights = weights;
        this.kernelSize = kernelSize;
        this.kernelDim = kernelDim;
        this.stride = stride;
        this.padding = padding;
        
        this.input = input;
        this.inputSize = inputSize;
        this.inputDim = inputDim;
        
        this.output = output;
        this.outputSize = outputSize;
        this.outputDim = outputDim;
        
        Range range = Range.create3D(outputSize[0], outputSize[1], outputSize[2] * outputSize[3]);
        execute(range);
    }
    
    private float getInput(int i, int j, int k, int n) {
        i = i - padding[0];
        j = j - padding[1];
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= inputSize[0] || j >= inputSize[1]) {
            return 0;
        }
        
        return input[n * inputDim[2] + k * inputDim[1] + j * inputDim[0] + i];
    }
    
    private float getWeight(int i, int j, int k, int n) {
        return weights[n * kernelDim[2] + k * kernelDim[1] + j * kernelDim[0] + i];
    }
    
    private float getBiasWeight(int n) {
        return weights[(n + 1) * kernelDim[2] - 1];
    }
    
    @Override
    public void run() {
        int oi = getGlobalId(0); //Output Volume i,j,n
        int oj = getGlobalId(1);
        int ok = getGlobalId(2)      % outputSize[2];
        int on =(getGlobalId(2) - ok) / outputSize[2];
        
        int outputIndex = on * outputDim[2] + ok * outputDim[1] + oj * outputDim[0] + oi;
        
        int inputPosi = oi * stride[0];
        int inputPosj = oj * stride[1];
        
        output[outputIndex] = 0;
        
        for (int ii = 0; ii < kernelSize[0]; ii++) {
            for (int ij = 0; ij < kernelSize[1]; ij++) {
                for (int ik = 0; ik < kernelSize[2]; ik++) {
                    output[outputIndex] = output[outputIndex] + (getInput(ii + inputPosi, ij + inputPosj, ik, on) * getWeight(ii, ij, ik, ok));
                }
            }
        }
        
        output[outputIndex] = output[outputIndex] + getBiasWeight(ok);
    }
}
