import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        ###### START CODE HERE ######
        self.img=io.imread("inputPS1Q3.jpg")
        self.fig, self.axs = plt.subplots(3,2)
        self.fig.set_size_inches(12, 10)
        ###### END CODE HERE ######
        pass
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        gray=np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        ###### END CODE HERE ######
        pass
    
        return gray
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        r=self.img[:,:,0]
        g=self.img[:,:,1]
        b=self.img[:,:,2]
        swapImg= np.dstack((g,r,b))
        self.axs[0][0].imshow(swapImg)
        ###### END CODE HERE ######
        pass
    
        return swapImg
    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg=self.rgb2gray(self.img)
        self.axs[0][1].imshow(grayImg, cmap=plt.get_cmap('gray'))
        ###### END CODE HERE ######
        pass
    
        return grayImg
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg=self.rgb2gray(self.img)
        negativeImg= 255-grayImg
        self.axs[1][0].imshow(negativeImg, cmap=plt.get_cmap('gray'))
        ###### END CODE HERE ######
        pass
    
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg=self.rgb2gray(self.img)
        mirrorImg=np.flip(grayImg, 1)
        self.axs[1][1].imshow(mirrorImg, cmap=plt.get_cmap('gray'))
        ###### END CODE HERE ######
        pass
    
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        grayImg=self.rgb2gray(self.img)
        mirrorImg=np.flip(grayImg, 1)
        avgImg=((grayImg+mirrorImg)/2).astype(int)
        self.axs[2][0].imshow(avgImg, cmap=plt.get_cmap('gray'))
        ###### END CODE HERE ######
        pass
    
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            addNoiseImg: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
        """
        
        ###### START CODE HERE ######
        #noise=np.random.randint(255, size=(self.img.shape[0],self.img.shape[1]))
        #np.save("noise.npy", noise)
        noise= (np.load('noise.npy')).astype(float)
        grayImg=self.rgb2gray(self.img)
        addNoiseImg=(noise+grayImg).astype(int)
        addNoiseImg=np.clip(addNoiseImg, 0, 255)
        self.axs[2][1].imshow(addNoiseImg, cmap=plt.get_cmap('gray'))
        ###### END CODE HERE ######
        pass
    
        return addNoiseImg
        
        
if __name__ == '__main__':
    
    p3 = Prob3()
    
    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    addNoiseImg = p3.prob_3_6()
    