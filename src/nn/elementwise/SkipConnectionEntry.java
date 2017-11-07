/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.elementwise;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import nn.Layer;


/**
 *
 * @author bowen
 */
public class SkipConnectionEntry implements Layer {
    
    
    private float[] input = new float[0];
    private float[] inputError = new float[0];
    private int inputSize[] = new int[0];
    private int inputLength = 0;
    
    private float[] output = new float[0];
    private float[] outputError = new float[0];
    
    
    
    private final List<SkipConnectionExit> exits = new LinkedList<>();
    private double[] skipError = new double[0];
    
    public SkipConnectionExit createExit() {
        SkipConnectionExit newExit = new SkipConnectionExit(this);
        exits.add(newExit);
        return newExit;
    }
    
    public void addSkipError(float[] outputError) {
        if (outputError.length != inputLength) {
            throw new IllegalArgumentException("Wrong output error array size.");
        }
        Arrays.parallelSetAll(skipError, i -> skipError[i] + outputError[i]);
    }
    
    @Override
    public float[] forward(float[] input) {
        if (input.length != inputLength) {
            throw new IllegalArgumentException("Wrong input array size.");
        }
        
        this.input = input;
        this.output = Arrays.copyOf(this.input, input.length);
        
        this.skipError = new double[input.length];
        for (SkipConnectionExit exit : exits) {
            exit.setSkipInput(this.output);
        }
        
        return this.output;
    }

    @Override
    public float[] backward(float[] outputError) {
        if (outputError.length != inputLength) {
            throw new IllegalArgumentException("Wrong output error array size.");
        }
        
        this.outputError = outputError;
        this.inputError = new float[outputError.length];
        
        for (int i=0; i<this.inputError.length; i++) {
            this.inputError[i] = this.outputError[i] + (float)skipError[i];
        }
        return this.inputError;
    }

    @Override
    public void setInputSize(int[] size) {
        if (size.length < 1) {
            throw new IllegalArgumentException("Illegal input dimension: " + size.length);
        }
        inputSize = size;
        int length = size[0];
        for (int i=1; i<inputSize.length; i++) {
            length *= size[i];
        }
        inputLength = length;
    }

    @Override
    public int[] getInputSize() {
        return inputSize;
    }

    @Override
    public int[] getOutputSize() {
        return inputSize;
    }

    public static class SkipConnectionExit implements Layer {
        
        private final SkipConnectionEntry entry;
        private float[] skipInput = new float[0];
        
        
        public SkipConnectionExit(SkipConnectionEntry entry) {
            this.entry = entry;
        }
        
        public void setSkipInput(float[] input) {
            this.skipInput = input;
        }
        
        @Override
        public float[] forward(float[] input) {
            
            if (input.length != skipInput.length) {
                throw new IllegalArgumentException("Input of wrong size!");
            }
            
            float[] newInput = Arrays.copyOf(input, input.length);
            
            for (int i=0; i<newInput.length; i++) {
                newInput[i] = newInput[i] + skipInput[i];
            }
            return newInput;
        }

        @Override
        public float[] backward(float[] outputError) {
            entry.addSkipError(outputError);
            return outputError;
        }

        @Override
        public void setInputSize(int[] size) {
            if (!Arrays.equals(size, entry.getInputSize())) {
                throw new IllegalArgumentException("Skip connection exit size is different than entry size.");
            }
        }

        @Override
        public int[] getInputSize() {
            return entry.getInputSize();
        }

        @Override
        public int[] getOutputSize() {
            return entry.getOutputSize();
        }
        
    }
    
}
