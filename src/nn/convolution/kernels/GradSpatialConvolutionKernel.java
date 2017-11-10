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
public class GradSpatialConvolutionKernel extends Kernel {

    private float[] weights = new float[0];
    private float[] gradients = new float[0];
    private int[] kernelSize = new int[4]; //Width, Height, Depth, Number
    private int[] kernelDim = new int[4]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] input = new float[0];
    private int inputSize[] = new int[4]; //Width, Height, Depth
    private int inputDim[] = new int[4]; //Width, Width * Height, Total Length
    
    private float[] outputError = new float[0];
    private int[] outputErrorSize = new int[4];
    private int[] outputErrorDim = new int[4]; //Width, Width * Height, Total Length

    public void call(float[] weights, float[] gradients, int[] kernelSize, int[] kernelDim, int[] stride, int padding[], float[] input, int[] inputSize, int[] inputDim, float[] outputError, int[] outputErrorSize, int[] outputErrorDim) {
        this.weights = weights;
        this.gradients = gradients;
        this.kernelSize = kernelSize;
        this.kernelDim = kernelDim;
        this.stride = stride;
        this.padding = padding;
        
        this.input = input;
        this.inputSize = inputSize;
        this.inputDim = inputDim;
        
        this.outputError = outputError;
        this.outputErrorSize = outputErrorSize;
        this.outputErrorDim = outputErrorDim;
        
        
        Range range = Range.create3D(kernelSize[0], kernelSize[1], (kernelSize[2] + 1) * kernelSize[3]);
        execute(range);
        //updateBias();
    }
    
    private float getFromInput(int i, int j, int k, int n) {
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= inputSize[0] || j >= inputSize[1]) {
            return 0;
        }
        
        return input[n * inputDim[2] + k * inputDim[1] + j * inputDim[0] + i];
    }
    
    
    private float getFromOutputError(int i, int j, int k, int n) {
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= outputErrorSize[0] || j >= outputErrorSize[1]) {
            return 0;
        }
        
        return outputError[n * outputErrorDim[2] + k * outputErrorDim[1] + j * outputErrorDim[0] + i];
    }
    
    private void addGradient(int wi, int wj, int wk, int wn, float value) {
        int index = wn * kernelDim[2] + wk * kernelDim[1] + wj * kernelDim[0] + wi;
        gradients[index] = gradients[index] + value;
    }
    private void addBiasGradient(int wn, float value) {
        int index = (wn + 1) * kernelDim[2] - 1;
        gradients[index] = gradients[index] + value;
    
    }
    
    @Override
    public void run() {
        int i = getGlobalId(0); //Kernel Volume i,j,k,n
        int j = getGlobalId(1);
        int k = getGlobalId(2) % (kernelSize[2] + 1);
        int n =(getGlobalId(2) - k) / (kernelSize[2] + 1);
        
        float grad = 0;
        
        if (k == kernelSize[2]) {
            
            for (int oi = 0; oi < outputErrorSize[0]; oi++) {
                for (int oj = 0; oj < outputErrorSize[1]; oj++) {
                    for (int on = 0; on < outputErrorSize[3]; on++) {
                        grad = grad + (getFromOutputError(oi, oj, n, on));
                    }
                }
            }
            //grad = grad * prop[0];
            grad = grad / (outputErrorSize[0] * outputErrorSize[1]);
            addBiasGradient(n, grad);
            
        } else {
            for (int oi = 0; oi < outputErrorSize[0]; oi++) {
                for (int oj = 0; oj < outputErrorSize[1]; oj++) {
                    for (int on = 0; on < outputErrorSize[3]; on++) {
                        int inputPosi = oi * stride[0] - padding[0] + i;
                        int inputPosj = oj * stride[1] - padding[1] + j;
                        int inputPosn = on;

                        grad = grad + (getFromInput(inputPosi, inputPosj, k, inputPosn) * getFromOutputError(oi, oj, n, on));
                    }
                }
            }
            //grad = grad * prop[0];
            grad = grad / (outputErrorSize[0] * outputErrorSize[1]);
            addGradient(i, j, k, n, grad);
        }
    }
}
