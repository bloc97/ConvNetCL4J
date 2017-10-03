/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package uconvnn;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 *
 * @author bowen
 */
public class BackwardSpatialConvolutionKernel extends Kernel {

    private float[] weights = new float[0];
    private final int[] kernelSize = new int[4]; //Width, Height, Depth, Number
    private final int[] kernelDim = new int[4]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private final int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private final int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] outputError = new float[0];
    private final int[] outputErrorSize = new int[3];
    private final int[] outputErrorDim = new int[3]; //Width, Width * Height, Total Length
    
    private float[] inputError = new float[0];
    private final int inputErrorSize[] = new int[3]; //Width, Height, Depth
    private final int inputErrorDim[] = new int[3]; //Width, Width * Height, Total Length
    
    public void setLayer(float[] weights, int w, int h , int d, int n, int shorz, int svert, int phorz, int pvert) {
        if (weights.length != w * h * d * n + n) {
            throw new IllegalArgumentException("Wrong layer or specified size.");
        }
        this.weights = weights;
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
    
    public void setOutputError(float[] error, int w, int h, int d) {
        if (d != kernelSize[3]) {
            throw new IllegalArgumentException("Wrong input or specified size.");
        }
        outputError = error;
        outputErrorSize[0] = w;
        outputErrorSize[1] = h;
        outputErrorSize[2] = d;
        
        outputErrorDim[0] = w;
        outputErrorDim[1] = w * h;
        outputErrorDim[2] = w * h * d;
        
        inputErrorSize[0] = (outputErrorSize[0] - 1) * stride[0] - (2 * padding[0]) + kernelSize[0];
        inputErrorSize[1] = (outputErrorSize[1] - 1) * stride[1] - (2 * padding[1]) + kernelSize[1];
        inputErrorSize[2] = kernelSize[2];
        
        inputErrorDim[0] = inputErrorSize[0];
        inputErrorDim[1] = inputErrorSize[0] * inputErrorSize[1];
        inputErrorDim[2] = inputErrorSize[0] * inputErrorSize[1] * inputErrorSize[2];
        inputError = new float[inputErrorDim[2]];
    }
    
    public float[] getInputError() {
        return inputError;
    }
    
    public float[] backward() {
        Range range = Range.create3D(inputErrorSize[0], inputErrorSize[1], inputErrorSize[2]);
        execute(range);
        return inputError;
    }
    
    public float getOutputError(int i, int j, int k, int wi, int wj, int wn) {
        
        int i_rel = i - wi + padding[0];
        int j_rel = j - wj + padding[1];
        
        if (i_rel % stride[0] == 0 && j_rel % stride[1] == 0) {
            return getFromOutputError(i_rel / stride[0], j_rel / stride[1], wn);
        }
        return 0;
    }
    
    public float getFromOutputError(int i, int j, int k) {
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= outputErrorSize[0] || j >= outputErrorSize[1]) {
            return 0;
        }
        
        return outputError[k * outputErrorDim[1] + j * outputErrorDim[0] + i];
    }
    
    public float getFromLayer(int i, int j, int k, int n) {
        return weights[n * kernelDim[2] + k * kernelDim[1] + j * kernelDim[0] + i];
    }
    
    public float getBiasFromLayer(int n) {
        return weights[(n + 1) * kernelDim[2] - 1];
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
                    inputError[inputErrorIndex] = inputError[inputErrorIndex] + getOutputError(i, j, k, wi, wj, wn);
                }
            }
        }
        
    }
}
