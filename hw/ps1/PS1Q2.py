import numpy as np
import matplotlib.pyplot as plt

class Prob2():
    def __init__(self):
        """Load inputAPS1Q2.npy here as a class variable A."""
        ###### START CODE HERE ######
        ###### END CODE HERE ######
        
    def prob_2_1(self):
        """Do plotting of intensities of A in decreasing value."""
        ###### START CODE HERE ######
        ###### END CODE HERE ######
    
    def prob_2_2(self):
        """Display histogram of A's intensities with 20 bins here."""
        ###### START CODE HERE ######
        ###### END CODE HERE ######
    
    def prob_2_3(self):
        """
        Create a new matrix X that consists of the bottom left quadrant of A here.
        Returns:
            X: bottom left quadrant of A which is of size 50 x 50
        """
        ###### START CODE HERE ######
        ###### END CODE HERE ######
    
        return X 
    
    def prob_2_4(self):
        """Create a new matrix Y, which is the same as A, but with A’s mean intensity value subtracted from each pixel.
        Returns:
            Y: A with A's mean intensity subtracted from each pixel. Output Y is of size 100 x 100.
        """
        ###### START CODE HERE ######
        ###### END CODE HERE ######
    
        return Y
    
    def prob_2_5(self):
        """
        Create your threshholded A i.e Z here.
        Returns:
            Z: A with only red pixels when the original value of the pixel is above the threshhold. Output Z is of size 100 x 100.
        """
        ###### START CODE HERE ######
        ###### END CODE HERE ######
    
        return Z
        
        
        
if __name__ == '__main__':
    
    p2 = Prob2()
    
    p2.prob_2_1()
    p2.prob_2_2()
    
    X = p2.prob_2_3()
    Y = p2.prob_2_4()
    Z = p2.prob_2_5()