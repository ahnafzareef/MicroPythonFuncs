#activation functions and layer math for prediction.
#[["ReLU","relu"],["ReLU6","relu6"],["Sigmoid","sigmoid"],["Tanh","tanh"],["Softmax","softmax"],["Linear","linear"],["ELU","elu"]];

import math
from random import random

def relu(x):
    return [max(0, y) for y in x]
def relu6(x):
    return [min(max(0, y), 6) for y in x]
def sigmoid(x):
    return [1 / (1 + math.exp(-y)) for y in x]
def tanh(x):
    return [math.tanh(y) for y in x]
def softmax(x):
    exp_x = [math.exp(y) for y in x]
    sum_exp_x = sum(exp_x)
    return [i / sum_exp_x for i in exp_x]
def linear(x):
    return x
def elu(x, alpha=1.0):
    return [y if y > 0 else alpha * (math.exp(y) - 1) for y in x]

def dense(inputs, weights, bias):
    #input frame comes in as 1D array, weights is 2D array, bias is 1D array
    #simply linear function y = mx + b. sum each node then add bias

    output = []
    for i in range(len(weights[0])):
        s = 0
        for j in range(len(inputs)):
            s += inputs[j] * weights[j][i] 
        s += bias[i] #add bias
        output.append(s) 

    return output

def flatten(x):
    return [item for sublist in x for item in (flatten(sublist) if isinstance(sublist, list) else [sublist])]

def batchNorm(input, gamma, beta, mean, variance, epsilon=0.001):

    #mean: sum of all inputs from 1 to size of batch, divided by size 
    #variance: sum of (input - mean)^2, divided by size
    #normalize: (input - mean) / sqrt(variance + epsilon) 
    #output: gamma * normalized + beta

    return [gamma[i] * (input[i] - mean[i]) / math.sqrt(variance[i] + epsilon) + beta[i] for i in range(len(input))]

def dropout(input, rate):
    #randomly sets a fraction of input units to 0 
    #rate is the fraction of the input units to drop, between 0 and 1.

    return [0 if random.random() < rate else x for x in input]

def conv2D(inputImage, filters, bias, padding, stride = 1):

    #conv: say 3x3 kernel, over a 3x3 image. multiplies 9 kernel elements w// all 9 pixes RIGHT UNDER IT. 
    #say img is [a,b,c][d,e,f][g,h,i] and kernel is [k1,k2,k3][k4,k5,k6][k7,k8,k9].
    #then output is axk1 + bxk2 + cxk3 + dxk4 + exk5 + fxk6 + gxk7 + hxk8 + ixk9 + bias. x is multply
    #and then the kernel window slides and repeats for the other pixels over and over
    #padding: "Valid" means no padding, "Same" means pad with zeros to keep same size
    #stride: how many pixels to move the kernel each time.

    #for conv2d, the input is 3 dimensions, [height, width, coloruChannels].
    #filters is 4D [numFilters, filterHeight, chnnaels, filters]
    #bias is 1D like before
    #the output is a conv2d model so it should be 3d [outputHeight, outputWidth, outChannel]

    inputHeight = len(inputImage)
    inputWidth = len(inputImage[0])
    inputChannels = len(inputImage[0][0])

    numFilters = len(filters[0][0][0])
    filterHeight = len(filters)
    filterWidth = len(filters[0])

    #Valid Padding
    outputHeightV = (inputHeight - filterHeight) // stride + 1 
    outputWidthV = (inputWidth - filterWidth) // stride + 1 
    
    outputHeightS = math.ceil(inputHeight / stride)
    outputWidthS = math.ceil(inputWidth / stride)
    
    # For every pixel, apply every filter, and for every filter multiply by pixels under + bias.
    
    if padding == "Valid":

        output = [[[0 for _ in range(numFilters)] for _ in range(outputWidthV)] for _ in range(outputHeightV)] #makes numFilters number of filters of dimensions width and height
        
        for x in range(0, outputHeightV):
            for y in range(0, outputWidthV):
                for f in range(numFilters):
                    sumOutput = 0;
                    for i in range(filterHeight):
                        for j in range(filterWidth):
                            for channels in range(inputChannels):
                                sumOutput += inputImage[x*stride+i][y*stride+j][channels] * filters[i][j][channels][f] #kernel; hieght, kernel width. in_channel. filters.
                    sumOutput += bias[f]
                    output[x][y][f] = sumOutput

    if padding == "Same":
        #same padding adds 0's around a point such that the filter can fit valid point of data within the kernel
        #check the overflow, add 0's if it goes out of bounds.

        padH = (filterHeight - 1) // 2
        padW = (filterWidth - 1) // 2
        
        paddedHeight = inputHeight + 2 * padH #org image + row on top and row on bottom
        paddedWidth = inputWidth + 2 * padW

        paddedImage = [[[0 for _ in range(inputChannels)] for _ in range(paddedWidth)] for _ in range(paddedHeight)] 

        #input copied to new image w/ pads
        for x in range(inputHeight):
            for y in range(inputWidth):
                for c in range(inputChannels):
                    paddedImage[x + padH][y + padW][c] = inputImage[x][y][c]

        output = [[[0 for _ in range(numFilters)] for _ in range(outputWidthS)] for _ in range(outputHeightS)]

        for x in range(outputHeightS):
            for y in range(outputWidthS):
                for f in range(numFilters):
                    sumOutput = 0
                    for i in range(filterHeight):
                        for j in range(filterWidth):
                            for c in range(inputChannels):
                                sumOutput += paddedImage[x*stride+i][y*stride+j][c] * filters[i][j][c][f]
                    sumOutput += bias[f]
                    output[x][y][f] = sumOutput
    return output

def maxPooling2D(inputImage, poolSize, stride):

    #slide window, take max.
    
    #in maxPooling they pad with -inf not 0 cus then 0 is never taken as max.

    inputHeight = len(inputImage)
    inputWidth = len(inputImage[0])
    inputChannels = len(inputImage[0][0])

    outputHeight = (inputHeight - poolSize) // stride + 1
    outputWidth = (inputWidth - poolSize) // stride + 1

    output = [[[0 for _ in range(inputChannels)] for _ in range(outputWidth)] for _ in range(outputHeight)]
    for x in range(outputHeight):
        for y in range(outputWidth):
            for c in range(inputChannels):
                maxVal = float('-inf')
                for i in range(poolSize):
                    for j in range(poolSize):
                        maxVal = max(maxVal, inputImage[x*stride+i][y*stride+j][c])
                output[x][y][c] = maxVal

    return output

def averagePooling2D(inputImage, poolSize, stride):
    #same thing but avg

    inputHeight = len(inputImage)
    inputWidth = len(inputImage[0])
    inputChannels = len(inputImage[0][0])
    outputHeight = (inputHeight - poolSize) // stride + 1
    outputWidth = (inputWidth - poolSize) // stride + 1
    output = [[[0 for _ in range(inputChannels)] for _ in range(outputWidth)] for _ in range(outputHeight)]
    for x in range(outputHeight):
        for y in range(outputWidth):
            for c in range(inputChannels):
                sumVal = 0
                for i in range(poolSize):
                    for j in range(poolSize):
                        sumVal += inputImage[x*stride+i][y*stride+j][c]
                output[x][y][c] = sumVal / (poolSize * poolSize)
    return output

def globalAveragePooling2D(inputImage):
    #takes average of each channel
    inputHeight = len(inputImage)
    inputWidth = len(inputImage[0])
    inputChannels = len(inputImage[0][0])

    output = [0 for _ in range(inputChannels)]
    for c in range(inputChannels):
        sumVal = 0
        for x in range(inputHeight):
            for y in range(inputWidth):
                sumVal += inputImage[x][y][c]
        output[c] = sumVal / (inputHeight * inputWidth)
    return output

def depthwiseConv2D(inputImage, filters, bias, padding, stride = 1):
    #same as conv2d but each filter is applied to only one channel. so if there are 3 channels and 3 filters, each filter is applied to a different channel.
    #depthMultiplier means each input channel makes depthMultiplier output channels. so 3 input ch and depthMult=4 means 12 output channels.
    #filters is 4D [filterHeight, filterWidth, inChannels, depthMultiplier]
    #bias is 1D length inChannels * depthMultiplier
    #the output is 3d [outputHeight, outputWidth, inChannels * depthMultiplier]

    inputHeight = len(inputImage)
    inputWidth = len(inputImage[0])
    inputChannels = len(inputImage[0][0])

    filterHeight = len(filters)
    filterWidth = len(filters[0])
    depthMultiplier = len(filters[0][0][0])
    outputChannels = inputChannels * depthMultiplier

    #Valid Padding
    outputHeightV = (inputHeight - filterHeight) // stride + 1 
    outputWidthV = (inputWidth - filterWidth) // stride + 1 
    
    outputHeightS = math.ceil(inputHeight / stride)
    outputWidthS = math.ceil(inputWidth / stride)

    if padding == "Valid":

        output = [[[0 for _ in range(outputChannels)] for _ in range(outputWidthV)] for _ in range(outputHeightV)]
        
        for x in range(0, outputHeightV):
            for y in range(0, outputWidthV):
                for c in range(inputChannels): #each channel independent
                    for m in range(depthMultiplier): #each input ch produces multiple output ch
                        sumOutput = 0;
                        for i in range(filterHeight):
                            for j in range(filterWidth):
                                sumOutput += inputImage[x*stride+i][y*stride+j][c] * filters[i][j][c][m]
                        outIndex = c * depthMultiplier + m #which output channel
                        sumOutput += bias[outIndex]
                        output[x][y][outIndex] = sumOutput

    if padding == "Same":
        #same padding adds 0's around a point such that the filter can fit valid point of data within the kernel

        padH = (filterHeight - 1) // 2
        padW = (filterWidth - 1) // 2
        
        paddedHeight = inputHeight + 2 * padH
        paddedWidth = inputWidth + 2 * padW

        paddedImage = [[[0 for _ in range(inputChannels)] for _ in range(paddedWidth)] for _ in range(paddedHeight)] 

        #input copied to new image w/ pads
        for x in range(inputHeight):
            for y in range(inputWidth):
                for c in range(inputChannels):
                    paddedImage[x + padH][y + padW][c] = inputImage[x][y][c]

        output = [[[0 for _ in range(outputChannels)] for _ in range(outputWidthS)] for _ in range(outputHeightS)]

        for x in range(outputHeightS):
            for y in range(outputWidthS):
                for c in range(inputChannels):
                    for m in range(depthMultiplier):
                        sumOutput = 0
                        for i in range(filterHeight):
                            for j in range(filterWidth):
                                sumOutput += paddedImage[x*stride+i][y*stride+j][c] * filters[i][j][c][m]
                        outIndex = c * depthMultiplier + m
                        sumOutput += bias[outIndex]
                        output[x][y][outIndex] = sumOutput
    return output

def seperableConv2D(inputImage, depthwiseFilters, pointwiseFilters, bias, padding, stride=1):
    #depthwise + conv2d
    
    depthwiseBias = [0 for _ in range(len(depthwiseFilters[0][0]) * len(depthwiseFilters[0][0][0]))]
    depthwiseOutput = depthwiseConv2D(inputImage, depthwiseFilters, depthwiseBias, padding, stride)
    return conv2D(depthwiseOutput, pointwiseFilters, bias, "Valid", 1)
