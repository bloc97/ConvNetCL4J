/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package other;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 *
 * @author bowen
 */
public class ForwardSpatialSingleDepthWiseConvolutionKernel extends Kernel { //Treats each depth of the kernel as individual convolutions, while using a 2D kernel without depth
    
    private float[] layer = new float[0];
    private final int[] kernelSize = new int[3]; //Width, Height, Depth, Number
    private final int[] kernelDim = new int[3]; //Width, Width * Height, Width * Height * Depth, Total Length
    
    private final int[] stride = new int[2]; //Stride: Horizontal, Vertical
    private final int[] padding = new int[2]; //Padding: Horizontal, Vertical
    
    private float[] input = new float[0];
    private final int inputSize[] = new int[3]; //Width, Height, Depth
    private final int inputDim[] = new int[3]; //Width, Width * Height, Total Length
    
    private float[] output = new float[0];
    private final int[] outputSize = new int[3];
    private final int[] outputDim = new int[3]; //Width, Width * Height, Total Length
    
    public void setLayer(float[] layer, int w, int h, int n, int shorz, int svert, int phorz, int pvert) {
        //if (layer.length != w * h * d * n + n) {
            //throw new IllegalArgumentException("Wrong layer or specified size.");
        //}
        this.layer = layer;
        kernelSize[0] = w;
        kernelSize[1] = h;
        kernelSize[3] = n;
        
        kernelDim[0] = w;
        kernelDim[1] = w * h;
        kernelDim[2] = w * h + 1; //1 array entry for bias
        kernelDim[3] = w * h * n + n; //n array entries for biases
        
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
        
        outputSize[0] = (inputSize[0] - kernelSize[0] + (2 * padding[0])) / stride[0] + 1;
        outputSize[1] = (inputSize[1] - kernelSize[1] + (2 * padding[1])) / stride[1] + 1;
        outputSize[2] = kernelSize[3] * d;
        
        outputDim[0] = outputSize[0];
        outputDim[1] = outputSize[0] * outputSize[1];
        outputDim[2] = outputSize[0] * outputSize[1] * outputSize[2];
        output = new float[outputSize[0] * outputSize[1] * outputSize[2]];
    }
    
    public float[] getOutput() {
        return output;
    }
    
    public void forward() {
        Range range = Range.create3D(outputSize[0], outputSize[1], outputSize[2]);
        execute(range);
    }
    
    public float getFromInput(int i, int j, int k) {
        i = i - padding[0];
        j = j - padding[1];
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= inputSize[0] || j >= inputSize[1]) {
            return 0;
        }
        
        return input[k * inputDim[1] + j * inputDim[0] + i];
    }
    
    public float getFromLayer(int i, int j, int k, int n) {
        return layer[n * kernelDim[2] + k * kernelDim[1] + j * kernelDim[0] + i];
    }
    
    public float getBiasFromLayer(int n) {
        return layer[(n + 1) * kernelDim[2] - 1];
    }
    
    @Override
    public void run() {
        int i = getGlobalId(0); //Output Volume i,j,n
        int j = getGlobalId(1);
        int n = getGlobalId(2);
        
        int outputIndex = n * outputDim[1] + j * outputDim[0] + i;
        
        int ik = n % kernelSize[2];
        
        int inputPosi = i * stride[0];
        int inputPosj = j * stride[1];
        
        output[outputIndex] = 0;
        
        for (int ii = 0; ii < kernelSize[0]; ii++) {
            for (int ij = 0; ij < kernelSize[1]; ij++) {
                output[outputIndex] = output[outputIndex] + (getFromInput(ii + inputPosi, ij + inputPosj, ik) * getFromLayer(ii, ij, ik, n));
            }
        }
        
        output[outputIndex] = output[outputIndex] + getBiasFromLayer(n);
    }
    
}
