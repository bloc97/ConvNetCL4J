/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

/**
 *
 * @author bowen
 */
public class Utils {
    
    
    public static void writeImage(BufferedImage imageBuffer, String path) throws IOException {
        ImageIO.write(imageBuffer, "png", new File(path));
    }
    
    public static BufferedImage readImage(String path) throws IOException {
        return ImageIO.read(new File(path));
    }
    
    public static float clamp(float value, float a, float b) {
        float min = Math.min(a, b);
        float max = Math.max(a, b);
        if (value < min) {
            return min;
        } else if (value > max) {
            return max;
        } else {
            return value;
        }
    }
    
    public static int clamp(int value, int a, int b) {
        int min = Math.min(a, b);
        int max = Math.max(a, b);
        if (value < min) {
            return min;
        } else if (value > max) {
            return max;
        } else {
            return value;
        }
    }
    
}
