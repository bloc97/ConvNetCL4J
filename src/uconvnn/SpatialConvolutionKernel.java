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
public class SpatialConvolutionKernel extends Kernel { //Sums the result of kernel depths, normal convolution
    
    private float[] weights = new float[0];
    private float[] gradients = new float[0];
    private final int[] kernelSize = new int[4]; //Width, Height, Depth, Number
    private final int[] kernelDim = new int[4]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private final int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private final int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] input = new float[0];
    private final int inputSize[] = new int[3]; //Width, Height, Depth
    private final int inputDim[] = new int[3]; //Width, Width * Height, Total Length
    
    private float[] output = new float[0];
    private final int[] outputSize = new int[3];
    private final int[] outputDim = new int[3]; //Width, Width * Height, Total Length
    
    private float[] outputError = new float[0];
    
    private float[] inputError = new float[0];
    
    private final float[] prop = new float[1];
    
    private final int[] step = new int[1];
    
    public void setWeight(float[] weights, int w, int h , int d, int n, int shorz, int svert, int phorz, int pvert) { //Allow non-odd kernels and individual side padding
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
        
        gradients = new float[weights.length];
        
    }
    
    public void setInput(float[] input, int w, int h, int d) {
        if (input.length != w * h * d || d != kernelSize[2]) {
            throw new IllegalArgumentException("Wrong input or specified size.");
        }
        this.input = input;
        inputError = new float[input.length];
        inputSize[0] = w;
        inputSize[1] = h;
        inputSize[2] = d;
        
        inputDim[0] = w;
        inputDim[1] = w * h;
        inputDim[2] = w * h * d;
        
        outputSize[0] = (inputSize[0] - kernelSize[0] + (2 * padding[0])) / stride[0] + 1;
        outputSize[1] = (inputSize[1] - kernelSize[1] + (2 * padding[1])) / stride[1] + 1;
        outputSize[2] = kernelSize[3];
        
        outputDim[0] = outputSize[0];
        outputDim[1] = outputSize[0] * outputSize[1];
        outputDim[2] = outputSize[0] * outputSize[1] * outputSize[2];
        output = new float[outputDim[2]];
    }
    
    public void setOutputError(float[] error) {
        if (error.length != output.length) {
            throw new IllegalArgumentException("Wrong input or specified size.");
        }
        outputError = error;
    }
    
    private float getWeight(int wi, int wj, int wk, int wn) {
        if (wi < 0 || wj < 0 || wk < 0 || wn < 0) {
            return 0;
        } else if (wi >= kernelSize[0] || wj >= kernelSize[1] || wk >= kernelSize[2] || wn >= kernelSize[3]) {
            return 0;
        }
        
        return weights[wn * kernelSize[2] + wk * kernelDim[1] + wj * kernelDim[0] + wi];
    }
    private float getBiasWeight(int wn) {
        return weights[(wn + 1) * kernelDim[2] - 1];
    }
    private void addGradient(int wi, int wj, int wk, int wn, float value) {
        int index = wn * kernelSize[2] + wk * kernelDim[1] + wj * kernelDim[0] + wi;
        gradients[index] = gradients[index] + value;
    }
    private void addBiasGradient(int n, float value) {
        int index = (n + 1) * kernelDim[2] - 1;
        gradients[index] = gradients[index] + value;
    }
    public void resetGradients() {
        gradients = new float[weights.length];
    }
    private float getInput(int ii, int ij, int ik) {
        ii = ii - padding[0];
        ij = ij - padding[1];
        
        if (ii < 0 || ij < 0) {
            return 0;
        } else if (ii >= inputSize[0] || ij >= inputSize[1]) {
            return 0;
        }
        
        return input[ik * inputDim[1] + ij * inputDim[0] + ii];
    }
    private float getOutputError(int oi, int oj, int ok) {
        if (oi < 0 || oj < 0) {
            return 0;
        } else if (oi >= outputSize[0] || oj >= outputSize[1]) {
            return 0;
        }
        
        return outputError[ok * outputDim[1] + oj * outputDim[0] + oi];
    }
    
    public int[] getInputSize() {
        return inputSize;
    }
    public int[] getOutputSize() {
        return outputSize;
    }
    public int[] getWeightSize() {
        return kernelSize;
    }
    
    public float[] getOutput() {
        return output;
    }
    public float[] getInputError() {
        return inputError;
    }
    public float[] getWeights() {
        return weights;
    }
    public float[] getGradients() {
        return gradients;
    }
    
    
    private float getOutputErrorFromInputAndWeights(int ii, int ij, int ik, int wi, int wj, int wn) {
        
        int i_rel = ii - wi + padding[0];
        int j_rel = ij - wj + padding[1];
        
        if ((i_rel % stride[0]) == 0 && (j_rel % stride[1]) == 0) {
            return getOutputError(i_rel / stride[0], j_rel / stride[1], wn);
        } else {
            return 0;
        }
    }
    
    public float[] forward() {
        step[0] = 0;
        Range range = Range.create3D(outputSize[0], outputSize[1], outputSize[2]);
        execute(range);
        return output;
    }
    
    public float[] backward() {
        step[0] = 1;
        Range range = Range.create3D(inputSize[0], inputSize[1], inputSize[2]);
        execute(range);
        return inputError;
    }
    
    public float[] grad(float learningRate) {
        step[0] = 2;
        prop[0] = learningRate;
        Range range = Range.create3D(kernelSize[0], kernelSize[1], kernelSize[2] * kernelSize[3]);
        execute(range);
        
        step[0] = 3;
        range = Range.create(kernelSize[3]);
        execute(range);
        //updateBias();
        return gradients;
    }
    
    private void step2() {
        int i = getGlobalId(0); //Input Volume i,j,k
        int j = getGlobalId(1);
        int k = getGlobalId(2);

        int inputErrorIndex = k * inputDim[1] + j * inputDim[0] + i;

        inputError[inputErrorIndex] = 0;

        for (int wi = 0; wi < kernelSize[0]; wi++) {
            for (int wj = 0; wj < kernelSize[1]; wj++) {
                for (int wn = 0; wn < kernelSize[3]; wn++) {
                    inputError[inputErrorIndex] = inputError[inputErrorIndex] + getOutputErrorFromInputAndWeights(i, j, k, wi, wj, wn) * getWeight(wi, wj, k, wn);
                }
            }
        }
        
    }
    
    @Override
    public void run() {
        int i, j, n, k = 0;
        if (step[0] == 0) {
            i = getGlobalId(0); //Output Volume i,j,n
            j = getGlobalId(1);
            n = getGlobalId(2);

            int outputIndex = n * outputDim[1] + j * outputDim[0] + i;

            int inputPosi = i * stride[0];
            int inputPosj = j * stride[1];

            output[outputIndex] = 0;

            for (int ii = 0; ii < kernelSize[0]; ii++) {
                for (int ij = 0; ij < kernelSize[1]; ij++) {
                    for (int ik = 0; ik < kernelSize[2]; ik++) {
                        output[outputIndex] = output[outputIndex] + (getInput(ii + inputPosi, ij + inputPosj, ik) * getWeight(ii, ij, ik, n));
                    }
                }
            }

            output[outputIndex] = output[outputIndex] + getBiasWeight(n);
        } else if (step[0] == 1) {
            step2();
        } /*else if (step[0] == 2) {
            i = getGlobalId(0); //Kernel Volume i,j,k,n
            j = getGlobalId(1);
            k = getGlobalId(2) % kernelSize[2];
            n =(getGlobalId(2) - k) % kernelSize[2];

            float grad = 0;

            for (int oi = 0; oi < outputSize[0]; oi++) {
                for (int oj = 0; oj < outputSize[1]; oj++) {
                    int inputPosi = oi * stride[0] + i;
                    int inputPosj = oj * stride[1] + j;

                    grad = grad + (getInput(inputPosi, inputPosj, k) * getOutputError(oi, oj, n));
                }
            }
            grad = grad * prop[0];
            grad = grad / (outputSize[0] * outputSize[1]);
            //addGradient(i, j, k, n, grad);
        } else if (step[0] == 3) {
            n = getGlobalId(0); //Kernel Bias n
            
            float grad = 0;
            
            for (int oi = 0; oi < outputSize[0]; oi++) {
                for (int oj = 0; oj < outputSize[1]; oj++) {
                    grad = grad + (getOutputError(oi, oj, n));
                }
            }
            grad = grad * prop[0];
            grad = grad / (outputSize[0] * outputSize[1]);
            addBiasGradient(n, grad);
        }*/
    }
    
}
