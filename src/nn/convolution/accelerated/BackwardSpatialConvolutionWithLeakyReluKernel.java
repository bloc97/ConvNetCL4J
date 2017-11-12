/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.convolution.accelerated;

import nn.convolution.kernels.*;
import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;

/**
 *
 * @author bowen
 */
public class BackwardSpatialConvolutionWithLeakyReluKernel extends Kernel {

    private float[] weights = new float[0];
    private int[] kernelSize = new int[4]; //Width, Height, Depth, Number
    private int[] kernelDim = new int[4]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] outputError = new float[0];
    private int[] outputErrorSize = new int[4];
    private int[] outputErrorDim = new int[4];
    
    private float[] inputError = new float[0];
    private int inputErrorSize[] = new int[4];
    private int inputErrorDim[] = new int[4];

    private float[] prop = new float[1];
    
    public void call(float[] weights, int[] kernelSize, int[] kernelDim, int[] stride, int padding[], float[] outputError, int[] outputErrorSize, int[] outputErrorDim, float[] inputError, int[] inputErrorSize, int[] inputErrorDim, float leakiness) {
        this.weights = weights;
        this.kernelSize = kernelSize;
        this.kernelDim = kernelDim;
        this.stride = stride;
        this.padding = padding;
        
        this.outputError = outputError;
        this.outputErrorSize = outputErrorSize;
        this.outputErrorDim = outputErrorDim;
        
        this.inputError = inputError;
        this.inputErrorSize = inputErrorSize;
        this.inputErrorDim = inputErrorDim;
        
        this.prop[0] = leakiness;
        
        Range range = Range.create3D(inputErrorSize[0], inputErrorSize[1], inputErrorSize[2] * inputErrorSize[3]);
        execute(range);
    }
    
    private float getOutputErrorByInputAndWeight(int i, int j, int k, int n, int wi, int wj, int wn) {
        
        int i_rel = i - wi + padding[0];
        int j_rel = j - wj + padding[1];
        
        if (i_rel % stride[0] == 0 && j_rel % stride[1] == 0) {
            return getOutputError(i_rel / stride[0], j_rel / stride[1], wn, n);
        }
        return 0;
    }
    
    private float getWeight(int wi, int wj, int wk, int wn) {
        
        if (wi < 0 || wj < 0 || wk < 0 || wn < 0) {
            return 0;
        } else if (wi >= kernelSize[0] || wj >= kernelSize[1] || wk >= kernelSize[2] || wn >= kernelSize[3]) {
            return 0;
        }
        
        return weights[wn * kernelDim[2] + wk * kernelDim[1] + wj * kernelDim[0] + wi];
    }
    
    private float getOutputError(int i, int j, int k, int n) {
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= outputErrorSize[0] || j >= outputErrorSize[1]) {
            return 0;
        }
        
        float tempOutputError = outputError[n * outputErrorDim[2] + k * outputErrorDim[1] + j * outputErrorDim[0] + i];
        
        
        if (tempOutputError < 0) {
            return prop[0] * tempOutputError;
        } else {
            return outputError[i];
        }
        
    }
    
    @Override
    public void run() {
        int i = getGlobalId(0); //Input Volume i,j,k
        int j = getGlobalId(1);
        int k = getGlobalId(2)      % inputErrorSize[2];
        int n =(getGlobalId(2) - k) / inputErrorSize[2];
        
        int inputErrorIndex = n * inputErrorDim[2] + k * inputErrorDim[1] + j * inputErrorDim[0] + i;
        
        inputError[inputErrorIndex] = 0;
        
        for (int wi = 0; wi < kernelSize[0]; wi++) {
            for (int wj = 0; wj < kernelSize[1]; wj++) {
                for (int wn = 0; wn < kernelSize[3]; wn++) {
                    inputError[inputErrorIndex] = inputError[inputErrorIndex] + (getOutputErrorByInputAndWeight(i, j, k, n, wi, wj, wn) * getWeight(wi, wj, k, wn));
                }
            }
        }
        
    }
}
