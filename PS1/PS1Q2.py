import numpy as np
import matplotlib.pyplot as plt

class Prob2():
    def __init__(self):
        """Load inputAPS1Q2.npy here as a class variable A."""
        ###### START CODE HERE ######
        self.A= np.load('inputAPS1Q2.npy')
        self.fig, self.axs = plt.subplots(2)
        self.fig.set_size_inches(7, 7)
        ###### END CODE HERE ######
        pass
        
    def prob_2_1(self):
        """Do plotting of intensities of A in decreasing value."""
        ###### START CODE HERE ######
        flat=self.A.flatten()
        sortedList=list(flat)
        sortedList.sort(reverse=True)
        arr=np.array([sortedList])
        width=arr.shape[1]
        height=arr.shape[1]/20
        self.axs[0].imshow(arr, cmap=plt.get_cmap('gray'), extent=(0, width, 0, height))
        self.axs[0].title.set_text("2.1 Intensity Plot")
        self.axs[0].get_yaxis().set_visible(False)
        ###### END CODE HERE ######
        pass
    
    def prob_2_2(self):
        """Display histogram of A's intensities with 20 bins here."""
        ###### START CODE HERE ######
        flat=self.A.flatten()
        fig=plt.hist(flat, bins=20, edgecolor='black', linewidth=1.2)
        plt.title('2.2 Intensity Histogram')
        plt.xlabel("Intensity Range")
        plt.ylabel("Frequency")
        plt.savefig("2_2.png")
        plt.show()
        ###### END CODE HERE ######
        pass
    
    def prob_2_3(self):
        """
        Create a new matrix X that consists of the bottom left quadrant of A here.
        Returns:
            X: bottom left quadrant of A which is of size 50 x 50
        """
        ###### START CODE HERE ######
        half_rows=int(self.A.shape[0]/2)
        half_cols=int(self.A.shape[1]/2)
        X=self.A[half_rows:,:half_cols]
        ###### END CODE HERE ######
        pass 
    
        return X 
    
    def prob_2_4(self):
        """Create a new matrix Y, which is the same as A, but with A’s mean intensity value subtracted from each pixel.
        Returns:
            Y: A with A's mean intensity subtracted from each pixel. Output Y is of size 100 x 100.
        """
        ###### START CODE HERE ######
        avg=self.A.mean()
        Y=self.A-avg
        ###### END CODE HERE ######
        pass
    
        return Y
    
    def prob_2_5(self):
        """
        Create your threshholded A i.e Z here.
        Returns:
            Z: A with only red pixels when the original value of the pixel is above the threshhold. Output Z is of size 100 x 100.
        """
        ###### START CODE HERE ######
        avg=self.A.mean()
        Z=self.A
        Z[Z > avg] = 1
        Z[Z <= avg] = 0
        bg=np.zeros((100,100))
        Z=np.dstack((Z,bg,bg))
        fig=plt.imshow(Z)
        plt.savefig("outputZPS1Q2.png")
        plt.close()
        ###### END CODE HERE ######
        pass
    
        return Z
        
        
        
if __name__ == '__main__':
    
    p2 = Prob2()
    
    p2.prob_2_1()
    p2.prob_2_2()
    
    X = p2.prob_2_3()
    Y = p2.prob_2_4()
    Z = p2.prob_2_5()