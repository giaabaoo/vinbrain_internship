import numpy as np
import pydicom

def getMaskAndImg(dataframe, idx):
    ds = pydicom.dcmread(dataframe["filepath"][idx])
    img = ds.pixel_array # img is 2-D matrix
    mask = run_length_decode_nguyen(dataframe["EncodedPixels"][idx]) # convert rle into mask
    img = np.stack((img, ) * 3, axis=-1) # get matrix shape (width, height, 3)
    return img, mask


def run_length_decode_nguyen(rle, height = 1024, width = 1024, fill_value = 1):
    if rle == "-1": 
      return np.zeros((height, width), np.float32) # negative case
    # init mask matrix
    component = np.zeros((height, width), np.float32)
    # flatten init matrix into list
    component = component.reshape(-1)
    # processing rle
    rle = rle[1: -1] # ignore character "[", "]"
    # get all value from rle, store in tempString
    tempString = []
    for eachRLE in rle.split(','):
      s = eachRLE.replace('\'', '')
      tempString.extend([int(s) for s in s.strip().split(" ")])
    # convert into numpy array  
    rle = np.asarray(tempString)
    # convert rle into mask
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start: end] = fill_value # value in rle is idx of pixel 1 in matrix we want
        start = end
    # reshape into mask matrix with shape (width, height)
    component = component.reshape(width, height).T
    return component

# without transpose
def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

# have to transpose the mask
def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)
