/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn;

/**
 *
 * @author bowen
 */
public interface Layer extends Bidirectional {
    
    public void setInputSize(int[] size);
    
    public int[] getInputSize();
    public int[] getOutputSize();
    
}
