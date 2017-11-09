/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package image;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

/**
 *
 * @author bowen
 */
public class ARGBImage {
    
    private final BufferedImage imageBuffer;
    private final float[] imageFloatExpanded3;
    private final float[] imageFloatExpanded4;
    
    private final int w, h;
    private final boolean hasAlpha;
    
    private final float min, max, range;
    
    public ARGBImage(BufferedImage image, float darkestValue, float brightestValue) {
        imageBuffer = image;
        min = darkestValue;
        max = brightestValue;
        range = max - min;
        
        w = imageBuffer.getWidth();
        h = imageBuffer.getHeight();
        hasAlpha = imageBuffer.getColorModel().hasAlpha();
        
        int[] imageInt = imageBuffer.getRGB(0, 0, w, h, null, 0, w);
        
        imageFloatExpanded3 = new float[w * h * 3];
        imageFloatExpanded4 = new float[w * h * 4];
        
        int hdim = w;
        int depthdim = w * h;
        //Implement this in parallel
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                
                int argb = imageInt[j * w + i];
                int a = argb >> 24 & 0xff;
                int r = argb >> 16 & 0xff;
                int g = argb >>  8 & 0xff;
                int b = argb       & 0xff;
                
                float alpha = (a / 255f) * range + min;
                float red = (r / 255f) * range + min;
                float green = (g / 255f) * range + min;
                float blue = (b / 255f) * range + min;
                
                imageFloatExpanded3[0 * depthdim + j * hdim + i] = red;
                imageFloatExpanded3[1 * depthdim + j * hdim + i] = green;
                imageFloatExpanded3[2 * depthdim + j * hdim + i] = blue;
                
                imageFloatExpanded4[0 * depthdim + j * hdim + i] = alpha;
                imageFloatExpanded4[1 * depthdim + j * hdim + i] = red;
                imageFloatExpanded4[2 * depthdim + j * hdim + i] = green;
                imageFloatExpanded4[3 * depthdim + j * hdim + i] = blue;
            }
        }
        
    }
    
    public ARGBImage(float[] image, int[] imageSize, float darkestValue, float brightestValue) {
        if (imageSize.length != 3) {
            if (imageSize.length != 4) {
                throw new IllegalArgumentException("Image must be dimension 3 or 4.");
            } else if (imageSize.length == 4 && imageSize[3] != 1) {
                throw new IllegalArgumentException("Cannot process a image batch into a single image.");
            }
        } else if (imageSize[2] != 3 && imageSize[2] != 4) {
            throw new IllegalArgumentException("Image must have 3 or 4 channels.");
        }
        
        w = imageSize[0];
        h = imageSize[1];
        int d = imageSize[2];
        hasAlpha = (d == 4);
        
        if (image.length != w * h * d) {
            throw new IllegalArgumentException("Image array does not match image size.");
        }
        
        min = darkestValue;
        max = brightestValue;
        range = max - min;
        
        int wh = w * h;
        
        if (hasAlpha) {
            imageFloatExpanded4 = Arrays.copyOf(image, image.length);
            imageFloatExpanded3 = Arrays.copyOfRange(image, wh, image.length);
        } else {
            imageFloatExpanded3 = Arrays.copyOf(image, image.length);
            imageFloatExpanded4 = new float[w * h * 4];
            System.arraycopy(image, 0, imageFloatExpanded4, wh, image.length);
            Arrays.fill(imageFloatExpanded4, 0, wh, brightestValue);
        }
        
        imageBuffer = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        
        //Implement this in parallel
        for (int i=0; i<w; i++) {
            for (int j=0; j<h; j++) {
                float alpha = imageFloatExpanded4[         j * w + i];
                float red   = imageFloatExpanded4[1 * wh + j * w + i];
                float green = imageFloatExpanded4[2 * wh + j * w + i];
                float blue  = imageFloatExpanded4[3 * wh + j * w + i];
                
                float normAlpha = (alpha - min) / range;
                float normRed   = (red   - min) / range;
                float normGreen = (green - min) / range;
                float normBlue  = (blue  - min) / range;
                
                int a = (int) (normAlpha * 255);
                int r = (int) (normRed   * 255);
                int g = (int) (normGreen * 255);
                int b = (int) (normBlue  * 255);
                
                a = Utils.clamp(a, 0, 255);
                r = Utils.clamp(r, 0, 255);
                g = Utils.clamp(g, 0, 255);
                b = Utils.clamp(b, 0, 255);
                
                int argb = a << 24 | r << 16 | g << 8 | b;
                
                imageBuffer.setRGB(i, j, argb);
            }
        }
        
    }
    
    public int getWidth() {
        return w;
    }

    public int getHeight() {
        return h;
    }

    public boolean hasAlpha() {
        return hasAlpha;
    }
    
    public int[] getImageSize(boolean useAlpha) {
        return new int[] {w, h, (useAlpha) ? 4 : 3};
    }

    public BufferedImage getBuffer() {
        return imageBuffer;
    }

    public float[] getFloatExpanded(boolean useAlpha) {
        if (useAlpha) {
            return Arrays.copyOf(imageFloatExpanded4, imageFloatExpanded4.length);
        } else {
            return Arrays.copyOf(imageFloatExpanded3, imageFloatExpanded3.length);
        }
    }
    
    
    
}
