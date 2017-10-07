/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn;

import nn.loss.MeanSquaredErrorLossFunction;
import nn.loss.LossFunction;
import nn.convolution.SpatialConvolutionLayer;
import com.aparapi.device.Device;
import com.aparapi.device.OpenCLDevice;
import java.awt.Color;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;
import nn.elementwise.activation.ReLULayer;
import nn.optim.SGD;
import other.Scalr;

/**
 *
 * @author bowen
 */
public class UConvNN {

    
    public static float[] wrapImage(int[] rgbArray, int w, int h) {
        
        float[] floatArray = new float[w*h*3];
        
        int hdim = w;
        int depthdim = w * h;
        
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                Color c = new Color(rgbArray[j * w + i]);
                float red = c.getRed() / 255f;
                float green = c.getGreen() / 255f;
                float blue = c.getBlue() / 255f;
                
                floatArray[0 * depthdim + j * hdim + i] = red;
                floatArray[1 * depthdim + j * hdim + i] = green;
                floatArray[2 * depthdim + j * hdim + i] = blue;
            }
        }
        
        return floatArray;
    }
    
    public static BufferedImage getImage(float[] outputArray, int w, int h) {
        
        BufferedImage image = new BufferedImage(h, h, BufferedImage.TYPE_INT_ARGB);
        
        int hdim = w;
        int depthdim = w * h;
        
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                int red = (int)((outputArray[0 * depthdim + j * hdim + i] + 1) * 127.5f);
                int green = (int)((outputArray[1 * depthdim + j * hdim + i] + 1) * 127.5f);
                int blue = (int)((outputArray[2 * depthdim + j * hdim + i] + 1) * 127.5f);
                
                red = clamp(red, 0, 255);
                green = clamp(green, 0, 255);
                blue = clamp(blue, 0, 255);
                
                Color c = new Color(red, green, blue, 255);
                
                image.setRGB(i, j, c.getRGB());
            }
        }
        
        return image;
        
    }
    public static BufferedImage getResidualImage(float[] outputArray, int w, int h) {
        
        BufferedImage image = new BufferedImage(h, h, BufferedImage.TYPE_INT_ARGB);
        
        int hdim = w;
        int depthdim = w * h;
        
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                int red = (int)((outputArray[0 * depthdim + j * hdim + i] + 1) * 127.5f);
                int green = (int)((outputArray[1 * depthdim + j * hdim + i] + 1) * 127.5f);
                int blue = (int)((outputArray[2 * depthdim + j * hdim + i] + 1) * 127.5f);
                
                red = clamp(red, 0, 255);
                green = clamp(green, 0, 255);
                blue = clamp(blue, 0, 255);
                
                Color c = new Color(red, green, blue, 255);
                
                image.setRGB(i, j, c.getRGB());
            }
        }
        
        return image;
        
    }
    
    public static int clamp(int value, int min, int max) {
        if (value > max) {
            return max;
        } else if (value < min) {
            return min;
        } else {
            return value;
        }
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        
        
        
        BufferedImage hr = ImageIO.read(new File("image.png"));
        BufferedImage small = Scalr.resize(hr, Scalr.Method.BALANCED, Scalr.Mode.FIT_EXACT, 50, 50);
        BufferedImage lr = Scalr.resize(small, Scalr.Method.BALANCED, Scalr.Mode.FIT_EXACT, 100, 100);
        
        int[] inputInt = lr.getRGB(0, 0, 100, 100, new int[100*100*3], 0, 100);
        int[] expectedOutputInt = hr.getRGB(0, 0, 100, 100, new int[100*100*3], 0, 100);
        
        Color c = new Color(expectedOutputInt[100*100]);
        
        System.out.println(c.getRed());
        System.out.println(c.getGreen());
        System.out.println(c.getBlue());
        
        
        float[] input = wrapImage(inputInt, 100, 100);
        float[] expectedOutput = wrapImage(expectedOutputInt, 100, 100);
        System.out.println(Arrays.toString(expectedOutputInt));
        System.out.println(Arrays.toString(expectedOutput));
        
        for (int i=0; i<input.length; i++) {
            expectedOutput[i] = (expectedOutput[i] - input[i]) * 2f;
            input[i] = input[i] * 2f - 1f;
        }
        
        BufferedImage image = getImage(input, 100, 100);
        ImageIO.write(image, "png", new File("input.png"));
        image = getImage(expectedOutput, 100, 100);
        ImageIO.write(image, "png", new File("expectedResid.png"));
        
        
        int x = 100;
        int y = 100;
        int d = 3;
        
        Network network = new Network();
        
        NeuronLayer nlayer = new SpatialConvolutionLayer(3, 3, d, 64, 1, 1, 1, 1);
        Randomiser.uniform(nlayer, 0, (float)Math.sqrt(6f/nlayer.getFanIn()));
        network.addLayer(nlayer);
        Layer alayer = new ReLULayer();
        network.addLayer(alayer);
        
        for (int i=0; i<5; i++) {
            nlayer = new SpatialConvolutionLayer(3, 3, 64, 64, 1, 1, 1, 1);
            Randomiser.uniform(nlayer, 0, (float)Math.sqrt(6f/nlayer.getFanIn()));
            network.addLayer(nlayer);
            alayer = new ReLULayer();
            network.addLayer(alayer);
        }
        
        nlayer = new SpatialConvolutionLayer(3, 3, 64, d, 1, 1, 1, 1);
        Randomiser.uniform(nlayer, 0, (float)Math.sqrt(6f/nlayer.getFanIn()));
        network.addLayer(nlayer);
        
        
        network.setInputSize(new int[] {x, y, d});
        
        LossFunction loss = new MeanSquaredErrorLossFunction();
        SGD sgd = new SGD();
        
        float[] output;
        float[] outputError;
        
        for (int i=0; i<1000; i++) {
            
            
            output = network.forward(input);
            outputError = loss.getErrorDerivativeArray(output, expectedOutput);
            network.backward(outputError);
            network.grad();
            
            
            float showError = loss.getError(output, expectedOutput);
            System.out.println(showError);
            
            if (i % 10 == 0) {
                image = getImage(output, 100, 100);
                ImageIO.write(image, "png", new File(i + "resid.png"));
            }
            
            sgd.update(network, 0.1f, 0.001f);
            
            
            
        }
        
        
    }
    
    private static void test() {
        
        
        
        
        System.out.println(OpenCLDevice.listDevices(Device.TYPE.GPU));
        System.out.println(OpenCLDevice.listDevices(Device.TYPE.CPU));
        
        int x = 100;
        int y = 100;
        int d = 3;
        
        Network network = new Network();
        
        NeuronLayer nlayer = new SpatialConvolutionLayer(3, 3, d, 64, 1, 1, 1, 1);
        Randomiser.uniform(nlayer, 0, (float)Math.sqrt(2f/nlayer.getFanIn()));
        network.addLayer(nlayer);
        Layer alayer = new ReLULayer();
        network.addLayer(alayer);
        
        for (int i=0; i<10; i++) {
            nlayer = new SpatialConvolutionLayer(3, 3, 64, 64, 1, 1, 1, 1);
            Randomiser.uniform(nlayer, 0, (float)Math.sqrt(2f/nlayer.getFanIn()));
            network.addLayer(nlayer);
            alayer = new ReLULayer();
            network.addLayer(alayer);
        }
        
        nlayer = new SpatialConvolutionLayer(3, 3, 64, d, 1, 1, 3, 3);
        Randomiser.uniform(nlayer, 0, (float)Math.sqrt(2f/nlayer.getFanIn()));
        network.addLayer(nlayer);
        alayer = new ReLULayer();
        //network.addLayer(alayer);
        
        
        float[] input1 = new float[x*y*d];
        float[] expectedOutput1 = new float[x*y*d];
        float[] input2 = new float[x*y*d];
        float[] expectedOutput2 = new float[x*y*d];
        float[] input3 = new float[x*y*d];
        float[] expectedOutput3 = new float[x*y*d];
        
        for (int i=0; i<input1.length; i++) {
            input1[i] = (float)(Math.random() * 2d - 1);
            input2[i] = (float)(Math.random() * 2d - 1);
            input3[i] = (float)(Math.random() * 2d - 1);
            expectedOutput1[i] = (float)(Math.random() * 2d - 1);
            expectedOutput2[i] = (float)(Math.random() * 2d - 1);
            expectedOutput3[i] = (float)(Math.random() * 2d - 1);

        }
        
        network.setInputSize(new int[] {x, y, d});
        
        LossFunction loss = new MeanSquaredErrorLossFunction();
        SGD sgd = new SGD();
        
        float[] output;
        float[] outputError;
        
        for (int i=0; i<1000; i++) {
            
            
            output = network.forward(input1);
            outputError = loss.getErrorDerivativeArray(output, expectedOutput1);
            network.backward(outputError);
            network.grad();
            
            output = network.forward(input2);
            outputError = loss.getErrorDerivativeArray(output, expectedOutput2);
            network.backward(outputError);
            network.grad();
            
            output = network.forward(input3);
            outputError = loss.getErrorDerivativeArray(output, expectedOutput3);
            network.backward(outputError);
            network.grad();
            
            float showError = loss.getError(output, expectedOutput3);
            System.out.println(showError);
            
            
            sgd.update(network, 0.1f, 0.01f);
            
            
            
        }
        /*
        
        
        
        
        int x = 14;
        int y = 14;
        
        SpatialConvolutionLayer layer = new SpatialConvolutionLayer(layerSize, layerSize, 3, 3, 1, 1, 1, 1);
        
        
        float[] weights = layer.getWeights();
        
        for (int i=0; i<weights.length; i++) {
            weights[i] = (float)(Math.random() * 2d - 1);
        }
        
        float[] input1 = new float[] {0.2f,0.5f,0.1f,0.4f,0.9f,0.5f,0.7f,0.6f,0.0f};
        float[] expectedOutput1 = new float[] {0.9f,0.5f,0.0f,0.4f,0.1f,0.7f,0.7f,0.6f,0.5f};
        float[] input2 = new float[] {0.2f,-0.5f,0.5f,0.3f,0.2f,-0.4f,0.5f,0.6f,0.0f};
        float[] expectedOutput2 = new float[] {-0.5f,0.52f,0.2f,0.4f,0.6f,0.5f,0.1f,0.3f,0.4f};
        float[] input3 = new float[] {-0.5f,0.1f,0.3f,0.2f,0.7f,-0.2f,-0.4f,0.4f,0.1f};
        float[] expectedOutput3 = new float[] {0.2f,0.0f,0.4f,0.9f,0.7f,0.3f,-0.5f,-0.6f,0.0f};
        
        
        float[] input1 = new float[x*y*3];
        float[] expectedOutput1 = new float[x*y*3];
        float[] input2 = new float[x*y*3];
        float[] expectedOutput2 = new float[x*y*3];
        float[] input3 = new float[x*y*3];
        float[] expectedOutput3 = new float[x*y*3];
        
        for (int i=0; i<input1.length; i++) {
            input1[i] = (float)(Math.random() * 2d - 1);
            input2[i] = (float)(Math.random() * 2d - 1);
            input3[i] = (float)(Math.random() * 2d - 1);
            expectedOutput1[i] = (float)(Math.random() * 2d - 1);
            expectedOutput2[i] = (float)(Math.random() * 2d - 1);
            expectedOutput3[i] = (float)(Math.random() * 2d - 1);

        }
        
        layer.setInputSize(new int[] {x, y, 3});
        
        float[] output = layer.forward(input1);
        float[] error = new float[output.length];
        
        for (int i=0; i<output.length; i++) {
            float diff = expectedOutput1[i] - output[i];
            error[i] = diff;
        }
        
        LossFunction lossFunction = new MeanSquaredErrorLossFunction();
        
        //System.out.println(Arrays.toString(input1));
        //System.out.println(Arrays.toString(expectedOutput1));
        //System.out.println(Arrays.toString(output));
        //System.out.println(Arrays.toString(error));
        
        float[] inputError = layer.backward(error);
        
        for (int i=0; i<100000; i++) {
            
            float totalError = 0;
            
            output = layer.forward(input1);
            error = lossFunction.getErrorDerivativeArray(output, expectedOutput1);
            inputError = layer.backward(error);
            layer.grad();
            totalError += lossFunction.getError(output, expectedOutput1);
            
            output = layer.forward(input2);
            error = lossFunction.getErrorDerivativeArray(output, expectedOutput2);
            inputError = layer.backward(error);
            layer.grad();
            totalError += lossFunction.getError(output, expectedOutput2);
            
            output = layer.forward(input3);
            error = lossFunction.getErrorDerivativeArray(output, expectedOutput3);
            inputError = layer.backward(error);
            layer.grad();
            totalError += lossFunction.getError(output, expectedOutput3);
            
            totalError /= 3;
            
            System.out.println("i: " + i + " " + totalError);
            
            float[] grad = layer.getGradients();
            
            for (int g=0; g<grad.length; g++) {
                weights[g] += grad[g] / 3 * 0.05f;
            }
            layer.resetGradients();
            //System.out.println(Arrays.toString(inputError));
            //System.out.println(Arrays.toString(grad));
            //System.out.println(Arrays.toString(error));
        }*/
    }
    
}
