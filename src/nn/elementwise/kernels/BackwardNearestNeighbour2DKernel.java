/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.elementwise.kernels;

import com.aparapi.Kernel;
import com.aparapi.Range;

/**
 *
 * @author bowen
 */
public class BackwardNearestNeighbour2DKernel extends Kernel {
    
    private float[] inputError = new float[0];
    private int inputErrorSize[] = new int[4]; //Width, Height, Depth
    private int inputErrorDim[] = new int[4]; //Width, Width * Height, Total Length
    
    private float[] outputError = new float[0];
    private int[] outputErrorSize = new int[4];
    private int[] outputErrorDim = new int[4]; //Width, Width * Height, Total Length

    public void call(float[] outputError, int[] outputErrorSize, int[] outputErrorDim, float[] inputError, int[] inputErrorSize, int[] inputErrorDim) {
        this.inputError = inputError;
        this.inputErrorSize = inputErrorSize;
        this.inputErrorDim = inputErrorDim;
        
        this.outputError = outputError;
        this.outputErrorSize = outputErrorSize;
        this.outputErrorDim = outputErrorDim;
        
        Range range = Range.create3D(inputErrorSize[0], inputErrorSize[1], inputErrorSize[2] * inputErrorSize[3]);
        execute(range);
    }
    
    private float getOutputError(int i, int j, int k, int n) {
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= outputErrorSize[0] || j >= outputErrorSize[1]) {
            return 0;
        }
        
        return outputError[n * outputErrorDim[2] + k * outputErrorDim[1] + j * outputErrorDim[0] + i];
    }
    
    @Override
    public void run() {
        int i = getGlobalId(0); //Output Volume i,j,n
        int j = getGlobalId(1);
        int k = getGlobalId(2) % inputErrorSize[2];
        int n = (getGlobalId(2) - k) / inputErrorSize[2];
        
        int inputErrorIndex = n * inputErrorDim[2] + k * inputErrorDim[1] + j * inputErrorDim[0] + i;
        
        float value = 0;
        
        for (int x=i*2; x<i*2+2; x++) {
            for (int y=j*2; y<j*2+2; y++) {
                value += getOutputError(x, y, k, n);
            }
        }
        inputError[inputErrorIndex] = value / 4;
        
    }
}
