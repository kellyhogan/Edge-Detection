import sys;
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndimg
from scipy import signal, misc
from numpy import linalg as LA
#np.set_printoptions(threshold=np.nan)   #this displays entire matrix

def getU2(x, y):
    u2 = (x**2) + (y**2)
    return u2

def getCmplxE_1(sigma, uDotE):
    #use 1j as an imaginary number because it is in the equation
    numerator = 1j*np.pi*uDotE
    denominator = 2.*sigma
    CmplxE_1 = np.exp(numerator/denominator)
    return CmplxE_1

def getCmplxE_2(u2, sigma):
    numerator = (-1.)*u2
    denominator = 2.*(sigma**2)
    CmplxE_2 = np.exp(numerator/denominator)
    return CmplxE_2

def getUDotE(x , y, theta):
    uDotE = (x*np.cos(theta))+(y*np.sin(theta))
    return uDotE 
    
def getWavelet(sigma, theta, xPixels, yPixels, xP, yP):
    #For C2
    numerator = 0
    denominator = 0
    for i in range(-xP, (xP+1)):
        for j in range(-yP, (yP+1)):
            cmplxE_1 = getCmplxE_1(sigma, getUDotE(i,j,theta))
            cmplxE_2 = getCmplxE_2(getU2(i,j), sigma)
            numerator += cmplxE_1 * cmplxE_2
            
            
    for i in range(-xP, (xP+1)):
        for j in range(-yP, (yP+1)):
            cmplxE_2 = getCmplxE_2(getU2(i,j), sigma)
            denominator += cmplxE_2       
            
    c2 = numerator/denominator
    #print(c2)
    
    #Now for C1
    c1_z = 0
    for i in range(-xP, (xP+1)):
        for j in range(-yP, (yP+1)):
            inner = ((np.pi)/(2*sigma))*(getUDotE(i,j,theta)) 
            theE = np.exp((-1*getU2(i,j))/(sigma**2))
            c1_z += (1 - 2*c2*np.cos(inner)+(c2**2))*theE
    c1 = sigma/np.sqrt(c1_z)
    #print(c1)
    
    #Make a blank grid to work with. use dtype=complex because dealing with complex numbers
    grid = np.zeros((xPixels,yPixels), dtype=complex)
    #Fill in grid
    
    for i in range(-xP, (xP+1)):
        #print("i = ", i)
        for j in range(-yP, (yP+1)):
            #print("j = ", j)
            cmplxE_1 = getCmplxE_1(sigma, getUDotE(i,j,theta))
            cmplxE_2 = getCmplxE_2(getU2(i,j), sigma)
            #Now actually plug into the original equation from HW question #1 
            grid[i+xP][j+yP] = (c1/sigma) * ((cmplxE_1 - c2)) * cmplxE_2
    
    return grid

def displayWavelets(sigV, thtV, waveArray):
    for k in range(0, sigV.size):
        for l in range(0, thtV.size):
   
            wavelet = getWavelet(sigV[k], thtV[l], 37, 37, 18, 18)
            print("Theta is : ", thtV[l], " Sigma is: ", sigV[k])
            waveArray.append(wavelet)
            displayImaginaryAndReal(wavelet);

def displayImaginaryAndReal(image):
    print("Imaginary:")
    plt.imshow(image.imag,cmap="gray")
    plt.show()
    print("Real:")
    plt.imshow(image.real,cmap="gray")
    plt.show()

def convolvePic(waveletArray, convArrayImag, convArrayReal, image):
    for wave in waveletArray:
        #needed to use signal.convolve2d() because these are both 2d matrices
        convolved = signal.convolve2d(wave, image)
        convArrayImag.append(convolved.imag)
        convArrayReal.append(convolved.real)
        displayImaginaryAndReal(convolved)

## for each convolution for imaginary, look at values for each pixel. 
##Pick |max| and add that to new grid. use that for histogram
def makeMaxForImaginary(xShape_2, yShape_2, convArray):
    #imaginaryMaxGrid = np.zeros((xShape_2, yShape_2))
    imaginaryMaxGrid = convArray[0]
    #for each convolution
    for conv in convArray:
        #loop through pixels and find value for each pixel and compare it to imaginaryMaxGrid
        for i in range(0, xShape_2):
            for j in range(0, yShape_2):
                imaginaryMaxGrid[i][j] = max(abs(imaginaryMaxGrid[i][j]), abs(conv[i][j]))
    return imaginaryMaxGrid

def makeMaxForReal(xShape_2, yShape_2, convArray):
    realMaxGrid = convArray[0]
    #for each convolution
    for conv in convArray:
        #loop through pixels and find value for each pixel and compare it to imaginaryMaxGrid
        for i in range(0, xShape_2):
            for j in range(0, yShape_2):
                realMaxGrid[i][j] = max(abs(realMaxGrid[i][j]), abs(conv[i][j]))
    return realMaxGrid

def makeMinForReal(xShape_2, yShape_2, convArray):
    realMinGrid = convArray[0]
    #for each convolution
    for conv in convArray:
        #loop through pixels and find value for each pixel and compare it to imaginaryMaxGrid
        for i in range(0, xShape_2):
            for j in range(0, yShape_2):
                realMinGrid[i][j] = min(abs(realMinGrid[i][j]), abs(conv[i][j]))
    return realMinGrid

def showPic(grid, msg):
    print(msg)
    plt.imshow(grid,cmap="gray")
    plt.show()
    return 

def getRatioU(imaginaryMaxGrid, realMaxGrid, epsilon):
    result = (realMaxGrid + (0.001 * epsilon))/(imaginaryMaxGrid + epsilon)
    #np.set_printoptions(threshold=np.nan)
    #print(result)
    return result

def getD(xShape, yShape, grid):
    maxRatio = grid[0][0]
    for i in range(0, xShape):
        for j in range(0, yShape):
            maxRatio = max(maxRatio, grid[i][j])
    #print("D is ", maxRatio)
    return maxRatio
            
def edgeWithRatio(ratioU, D):
    result = 1 - (ratioU/D)
    return result

def getC(xShape, yShape, imaginaryMaxGrid):
    maxRatio = imaginaryMaxGrid[0][0]
    for i in range(0, xShape):
        for j in range(0, yShape):
            maxRatio = max(maxRatio, imaginaryMaxGrid[i][j])
    return maxRatio

def getDiffU(beta, C, imaginaryMaxGrid, realMaxGrid, xShape, yShape, subResultA):
    Diff = beta*(imaginaryMaxGrid - C)
    for i in range(0, xShape):
        for j in range(0, yShape):
            subResultA[i][j] = realMaxGrid[i][j] - Diff[i][j]
    return

#the formula seemed to have changed
def getDiffU_noAlphaBeta(C, imaginaryMaxGrid, realMaxGrid, xShape, yShape, subResultA):
    #Diff = beta*(imaginaryMaxGrid - C)
    for i in range(0, xShape):
        for j in range(0, yShape):
            subResultA[i][j] = imaginaryMaxGrid[i][j] - realMaxGrid[i][j]
    return

def edgeWithDiff(diffU, alpha):
    result = np.exp(-1*alpha*diffU)
    return result

def edgeWithDiff_noAlphaBeta(diffU):
    result = np.exp(-1*diffU)
    return result

def showEdgeWithRatio(pupImagMax, pupRealMax, xShape_pup, yShape_pup, epsilon):
    ratio = getRatioU(pupImagMax, pupRealMax, epsilon)
    D = getD(xShape_pup, yShape_pup, ratio)
    edge_1 = edgeWithRatio(ratio, D)
    print(len(edge_1))
    print("ratio edge: epsilon = ", epsilon)
    plt.imshow(edge_1, cmap="gray")
    plt.show()

def getEdgeDiffMatrix(pupImagMax, pupRealMax, xShape_pup, yShape_pup, subResArray, alpha, beta):
    c_pup = getC(xShape_pup, yShape_pup, pupImagMax)
    getDiffU(beta, c_pup, pupImagMax, pupRealMax, xShape_pup, yShape_pup, subResArray)
    edge_2 = edgeWithDiff(subResArray, alpha)
    return edge_2

def showEdgeWithDiff(edge):
    #print("edge_2: beta = ", beta, "alpha = ", alpha)
    plt.imshow(edge, cmap="gray")
    plt.show()

######### End of functions #########
waveletArray = []
sigmaVals = np.array([2])
thetaVals = np.array([0, np.pi/4, np.pi/2, np.pi*3/4 ])

print("Displaying wavelets")
displayWavelets(sigmaVals, thetaVals, waveletArray)
image = cv2.imread(sys.argv[1], 0);
plt.imshow(image,cmap="gray")
plt.show()

convolvedArrayImag = []
convolvedArrayReal = []
print("Displaying convolutions")
convolvePic(waveletArray, convolvedArrayImag, convolvedArrayReal, image)

print("Displaying Max and Min Images")
xShape = convolvedArrayImag[0].shape[0]
yShape = convolvedArrayImag[0].shape[1]

imaginaryMaxGrid = makeMaxForImaginary(xShape, yShape, convolvedArrayImag)
showPic(imaginaryMaxGrid, "Imaginary:")

realMaxGrid = makeMaxForReal(xShape, yShape, convolvedArrayReal)
showPic(realMaxGrid, "Real Max:")

realMinGrid = makeMinForReal(xShape, yShape, convolvedArrayReal)
showPic(realMinGrid, "Real Min:")

print("edges with diff formula")
subResultArrayLeft = np.zeros((xShape, yShape))
edge = getEdgeDiffMatrix(imaginaryMaxGrid, realMaxGrid, xShape, yShape, subResultArrayLeft, 0.0000001, 1)
showEdgeWithDiff(edge)
