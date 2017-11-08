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
public class ForwardNearestNeighbour2DKernel extends Kernel {
    
    private float[] input = new float[0];
    private int inputSize[] = new int[3]; //Width, Height, Depth
    private int inputDim[] = new int[3]; //Width, Width * Height, Total Length
    
    private float[] output = new float[0];
    private int[] outputSize = new int[3];
    private int[] outputDim = new int[3]; //Width, Width * Height, Total Length

    public void call(float[] input, int[] inputSize, int[] inputDim, float[] output, int[] outputSize, int[] outputDim) {
        this.input = input;
        this.inputSize = inputSize;
        this.inputDim = inputDim;
        
        this.output = output;
        this.outputSize = outputSize;
        this.outputDim = outputDim;
        
        Range range = Range.create3D(outputSize[0], outputSize[1], outputSize[2]);
        execute(range);
    }
    
    private float getInput(int i, int j, int k) {
        
        if (i < 0 || j < 0) {
            return 0;
        } else if (i >= inputSize[0] || j >= inputSize[1]) {
            return 0;
        }
        
        return input[k * inputDim[1] + j * inputDim[0] + i];
    }
    
    @Override
    public void run() {
        int i = getGlobalId(0); //Output Volume i,j,n
        int j = getGlobalId(1);
        int k = getGlobalId(2);
        
        int outputIndex = k * outputDim[1] + j * outputDim[0] + i;
        
        output[outputIndex] = getInput(i/2, j/2, k);
    }
}
