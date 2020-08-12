import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""
        ###### START CODE HERE ######
        self.indoor=io.imread("indoor.png")
        self.outdoor=io.imread("outdoor.png")
        #fig, (ax1, ax2) = plt.subplots(2)
        #ax1.imshow(self.indoor)
        #ax2.imshow(self.outdoor)

        ###### END CODE HERE ######
        pass
    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        ###### START CODE HERE ######
        lab_in = cv2.cvtColor(self.indoor, cv2.COLOR_RGB2LAB)
        lab_out = cv2.cvtColor(self.outdoor, cv2.COLOR_RGB2LAB)
        #fig, (ax1, ax2) = plt.subplots(2)
        #ax1.imshow(lab_in)
        #ax2.imshow(lab_out)
        
        in_r=self.indoor[:,:,0]
        in_g=self.indoor[:,:,1]
        in_b=self.indoor[:,:,2]
        out_r=self.outdoor[:,:,0]
        out_g=self.outdoor[:,:,1]
        out_b=self.outdoor[:,:,2]
        
        lab_in_r=lab_in[:,:,0]
        lab_in_g=lab_in[:,:,1]
        lab_in_b=lab_in[:,:,2]
        lab_out_r=lab_out[:,:,0]
        lab_out_g=lab_out[:,:,1]
        lab_out_b=lab_out[:,:,2]
        
        fig, axs = plt.subplots(4, 3)
        fig.set_size_inches(15, 15)
        axs[0][0].imshow(in_r, cmap=plt.get_cmap('gray'))
        axs[0][1].imshow(in_g, cmap=plt.get_cmap('gray'))
        axs[0][2].imshow(in_b, cmap=plt.get_cmap('gray'))
        axs[1][0].imshow(out_r, cmap=plt.get_cmap('gray'))
        axs[1][1].imshow(out_g, cmap=plt.get_cmap('gray'))
        axs[1][2].imshow(out_b, cmap=plt.get_cmap('gray'))
        axs[2][0].imshow(lab_in_r, cmap=plt.get_cmap('gray'))
        axs[2][1].imshow(lab_in_g, cmap=plt.get_cmap('gray'))
        axs[2][2].imshow(lab_in_b, cmap=plt.get_cmap('gray'))
        axs[3][0].imshow(lab_out_r, cmap=plt.get_cmap('gray'))
        axs[3][1].imshow(lab_out_g, cmap=plt.get_cmap('gray'))
        axs[3][2].imshow(lab_out_b, cmap=plt.get_cmap('gray'))
        
        axs[0][0].title.set_text('indoor_rgb_r')
        axs[0][1].title.set_text('indoor_rbg_g')
        axs[0][2].title.set_text('indoor_rbg_b')
        axs[1][0].title.set_text('outdoor_rbg_r')
        axs[1][1].title.set_text('outdoor_rbg_g')
        axs[1][2].title.set_text('outdoor_rbg_b')
        axs[2][0].title.set_text('indoor_lab_r')
        axs[2][1].title.set_text('indoor_lab_g')
        axs[2][2].title.set_text('indoor_lab_b')
        axs[3][0].title.set_text('outdoor_lab_r')
        axs[3][1].title.set_text('outdoor_lab_g')
        axs[3][2].title.set_text('outdoor_lab_b')
        
        #plt.savefig("4_1.png")
        #plt.close()
        
        ###### END CODE HERE ######
        pass

    def prob_4_3(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        
        img = io.imread('inputPS1Q4.jpg') 
        img = img / 255.0
        
        ###### START CODE HERE ######
        #plt.imshow(img)
        HSV= np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rgb=img[i,j]
                r=rgb[0]
                g=rgb[1]
                b=rgb[2]
                max_index=rgb.argmax()
                V=np.max(rgb)
                m=np.min(rgb)
                C=V-m
                S=0
                Hprime=0
                H=0
                if(V!=0):
                    S=C/V
                    
                if(C==0):
                    HSV[i][j]=[H,S,V]
                    continue
                elif(max_index==0):
                    Hprime=(g-b)/C
                elif(max_index==1):
                    Hprime=((b-r)/C)+2
                elif(max_index==2):
                    Hprime=((r-g)/C)+4
                
                if(Hprime<0):
                    H=(Hprime/6)+1
                else:
                    H=(Hprime/6)
                
                HSV[i,j]=[H,S,V]
        
        plt.show()
        plt.imshow(HSV)
        plt.savefig("outputPS1Q4.png")
        plt.close()
        ###### END CODE HERE ######
        pass
    
        return HSV
        

        
if __name__ == '__main__':
    
    p4 = Prob4()
    
    p4.prob_4_1()

    HSV = p4.prob_4_3()

