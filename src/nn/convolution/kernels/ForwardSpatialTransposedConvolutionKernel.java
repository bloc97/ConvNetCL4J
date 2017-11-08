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
public class ForwardSpatialTransposedConvolutionKernel extends Kernel {

    private float[] weights = new float[0];
    private int[] kernelSize = new int[4]; //Width, Height, Depth, Number
    private int[] kernelDim = new int[4]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] outputError = new float[0];
    private int[] outputErrorSize = new int[3];
    private int[] outputErrorDim = new int[3]; //Width, Width * Height, Total Length
    
    private float[] inputError = new float[0];
    private int inputErrorSize[] = new int[3]; //Width, Height, Depth
    private int inputErrorDim[] = new int[3]; //Width, Width * Height, Total Length
    
    
    public void call(float[] weights, int[] kernelSize, int[] kernelDim, int[] stride, int padding[], float[] outputError, int[] outputErrorSize, int[] outputErrorDim, float[] inputError, int[] inputErrorSize, int[] inputErrorDim) {
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
        
        Range range = Range.create3D(inputErrorSize[0], inputErrorSize[1], inputErrorSize[2]);
        execute(range);
    }
    
    private float getOutputErrorByInputAndWeight(int i, int j, int k, int wi, int wj, int wn) {
        
        int i_rel = i - wi + padding[0];
        int j_rel = j - wj + padding[1];
        
        if (i_rel % stride[0] == 0 && j_rel % stride[1] == 0) {
            return getOutputError(i_rel / stride[0], j_rel / stride[1], wn);
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
    
    private float getBiasWeight(int n) {
        return weights[(n + 1) * kernelDim[2] - 1];
    }
    
    private float getOutputError(int i, int j, int k) {
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= outputErrorSize[0] || j >= outputErrorSize[1]) {
            return 0;
        }
        
        return outputError[k * outputErrorDim[1] + j * outputErrorDim[0] + i];
    }
    
    @Override
    public void run() {
        int i = getGlobalId(0); //Input Volume i,j,k
        int j = getGlobalId(1);
        int k = getGlobalId(2);
        
        int inputErrorIndex = k * inputErrorDim[1] + j * inputErrorDim[0] + i;
        
        inputError[inputErrorIndex] = 0;
        
        for (int wi = 0; wi < kernelSize[0]; wi++) {
            for (int wj = 0; wj < kernelSize[1]; wj++) {
                for (int wn = 0; wn < kernelSize[3]; wn++) {
                    inputError[inputErrorIndex] = inputError[inputErrorIndex] + (getOutputErrorByInputAndWeight(i, j, k, wi, wj, wn) * getWeight(wi, wj, k, wn));
                }
            }
        }
        for (int wn = 0; wn < kernelSize[3]; wn++) {
            inputError[inputErrorIndex] = inputError[inputErrorIndex] + getBiasWeight(wn);
        }
        
    }
}
