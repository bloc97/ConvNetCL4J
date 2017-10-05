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
public class UpdateSpatialConvolutionKernel extends Kernel {

    private float[] weights = new float[0];
    private final int[] kernelSize = new int[4]; //Width, Height, Depth, Number
    private final int[] kernelDim = new int[4]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private final int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private final int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] input = new float[0];
    private final int inputSize[] = new int[3]; //Width, Height, Depth
    private final int inputDim[] = new int[3]; //Width, Width * Height, Total Length
    
    private float[] outputError = new float[0];
    private final int[] outputErrorSize = new int[3];
    private final int[] outputErrorDim = new int[3]; //Width, Width * Height, Total Length
    
    private float[] prop = new float[1];
    
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
    
    public void setInput(float[] input, int w, int h, int d) {
        if (input.length != w * h * d || d != kernelSize[2]) {
            throw new IllegalArgumentException("Wrong input or specified size.");
        }
        this.input = input;
        inputSize[0] = w;
        inputSize[1] = h;
        inputSize[2] = d;
        
        inputDim[0] = w;
        inputDim[1] = w * h;
        inputDim[2] = w * h * d;
        
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
        
    }
    
    public void update(float learningRate) {
        Range range = Range.create3D(kernelSize[0], kernelSize[1], kernelSize[2] * kernelSize[3]);
        prop[0] = learningRate;
        execute(range);
        updateBias();
    }
    
    private float getWeight(int wi, int wj, int wk, int wn) {
        
        if (wi < 0 || wj < 0 || wk < 0 || wn < 0) {
            return 0;
        } else if (wi >= kernelSize[0] || wj >= kernelSize[1] || wk >= kernelSize[2] || wn >= kernelSize[3]) {
            return 0;
        }
        
        return weights[wn * kernelSize[2] + wk * kernelDim[1] + wj * kernelDim[0] + wi];
    }
    private void setWeight(int wi, int wj, int wk, int wn, float value) {
        
        if (wi < 0 || wj < 0 || wk < 0 || wn < 0) {
            return;
        } else if (wi >= kernelSize[0] || wj >= kernelSize[1] || wk >= kernelSize[2] || wn >= kernelSize[3]) {
            return;
        }
        
        weights[wn * kernelSize[2] + wk * kernelDim[1] + wj * kernelDim[0] + wi] = value;
    }
    
    private float getBiasInLayer(int n) {
        return weights[(n + 1) * kernelDim[2] - 1];
    }
    
    private void setBiasInLayer(int n, float value) {
        weights[(n + 1) * kernelDim[2] - 1] = value;
    }
    
    private float getFromInput(int i, int j, int k) {
        i = i - padding[0];
        j = j - padding[1];
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= inputSize[0] || j >= inputSize[1]) {
            return 0;
        }
        
        return input[k * inputDim[1] + j * inputDim[0] + i];
    }
    
    
    private float getFromOutputError(int i, int j, int k) {
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= outputErrorSize[0] || j >= outputErrorSize[1]) {
            return 0;
        }
        
        return outputError[k * outputErrorDim[1] + j * outputErrorDim[0] + i];
    }
    
    @Override
    public void run() {
        int i = getGlobalId(0); //Kernel Volume i,j,k,n
        int j = getGlobalId(1);
        int k = getGlobalId(2) % kernelSize[2];
        int n =(getGlobalId(2) - k) % kernelSize[2];
        
        float grad = 0;
        
        for (int oi = 0; oi < outputErrorSize[0]; oi++) {
            for (int oj = 0; oj < outputErrorSize[1]; oj++) {
                int inputPosi = oi * stride[0] + i;
                int inputPosj = oj * stride[1] + j;
                
                grad = grad + (prop[0] * getFromInput(inputPosi, inputPosj, k) * getFromOutputError(oi, oj, n));
            }
        }
        
        grad = grad / (outputErrorSize[0] * outputErrorSize[1]);
        
        setWeight(i, j, k, n, getWeight(i, j, k, n) + grad);
        
    }
    
    private void updateBias() {
        for (int n=0; n<kernelSize[3]; n++) {
            float grad = 0;
            
            for (int oi = 0; oi < outputErrorSize[0]; oi++) {
                for (int oj = 0; oj < outputErrorSize[1]; oj++) {
                    grad = grad + (prop[0] * getFromOutputError(oi, oj, n));
                }
            }
            grad = grad / (outputErrorSize[0] * outputErrorSize[1]);
            setBiasInLayer(n, getBiasInLayer(n) + grad);
        }
        
        
        
    }
}
