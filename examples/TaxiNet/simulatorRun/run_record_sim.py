import time
import numpy as np
import h5py
import mss
from PIL import Image
import cv2

import xpc3
import xpc3_helper

from nnet import *


# Read in the network
filename = "./TinyTaxiNet.nnet"
network = NNet(filename)

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
stride = 16             # Size of square of pixels downsampled to one grayscale value
# During downsampling, average the numPix brightest pixels in each square
numPix = 16
width = 256//stride    # Width of downsampled grayscale image
height = 128//stride    # Height of downsampled grayscale image

screenShot = mss.mss()
print(screenShot.monitors)
screen_width = 360  # For cropping
screen_height = 200  # For cropping

def noisy(noise_typ,image):

    if len(image.shape) == 3: 

        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 3.0
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy
        
    else:

        if noise_typ == "gauss":
            row,col= image.shape
            mean = 0
            var = 50.0
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col))
            gauss = gauss.reshape(row,col)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row,col = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col = image.shape
            gauss = np.random.randn(row,col)
            gauss = gauss.reshape(row,col)        
            noisy = image + image * gauss
            return noisy

def getCurrentImage():
    """ Returns a downsampled image of the current X-Plane 11 image
        compatible with the TinyTaxiNet neural network state estimator

        NOTE: this is designed for screens with 1920x1080 resolution
        operating X-Plane 11 in full screen mode - it will need to be adjusted
        for other resolutions
    """
    # Get current screenshot
    img = cv2.cvtColor(np.array(screenShot.grab(screenShot.monitors[2])),
                       cv2.COLOR_BGRA2BGR)[230:, :, :]
    img = cv2.resize(img, (screen_width, screen_height))
    img = img[:, :, ::-1]
    img = np.array(img)

    # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so
    # values range between 0 and 1
    imgNorm = np.array(Image.fromarray(img).convert('L').crop(
        (55, 5, 360, 135)).resize((256, 128)))/255.0

    # Downsample image
    # Split image into stride x stride boxes, average numPix brightest pixels in that box
    # As a result, img2 has one value for every box
    img2 = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            img2[i, j] = np.mean(np.sort(
                imgNorm[stride*i:stride*(i+1), stride*j:stride*(j+1)].reshape(-1))[-numPix:])


    # Ensure that the mean of the image is 0.5 and that values range between 0 and 1
    # The training data only contains images from sunny, 9am conditions.
    # Biasing the image helps the network generalize to different lighting conditions (cloudy, noon, etc)
    img2 -= img2.mean()
    img2 += 0.5
    img2[img2 > 1] = 1
    img2[img2 < 0] = 0

    img2 = noisy('gauss',img2*255.0)/255.0

    # Filename
    filename = 'savedImage.jpg'

    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename, img2*255.0)

    return img2.flatten(), img2, imgNorm

def getState(client):
    """ Returns an estimate of the crosstrack error (meters)
        and heading error (degrees) by passing the current
        image through TinyTaxiNet

        Args:
            client: XPlane Client
    """
    imageNN, downImg, upImg = getCurrentImage()
    pred = network.evaluate_network(imageNN)
    return pred[0], pred[1], downImg, upImg

def getControl(client, cte, he):
    """ Returns rudder command using proportional control
        for use with X-Plane 11 dynamics

        Args:
            client: XPlane Client
            cte: current estimate of the crosstrack error (meters)
            he: current estimate of the heading error (degrees)
    """
    # Amount of rudder needed to go straight, roughly.
    rudder = 0.008  # 0.004

    # Proportional controller
    cteGain = 0.015
    heGain = 0.008
    rudder += np.clip(cteGain * cte + heGain * he, -1.0, 1.0)
    return rudder

def run_sim(client, startCTE, startHE, startDTP, endDTP, runs, simSpeed=2.0, ToD=9.0,CC=0,logControl=True,saveFile='TaxiRun'):

    if logControl: 

        f = h5py.File(saveFile + '.h5', 'w')


    for i in np.arange(0,runs):
        # Time of day in local time, e.g. 8.0 = 8AM, 17.0 = 5PM
        client.sendDREF("sim/time/zulu_time_sec", ToD * 3600 + 8 * 3600)

        # Cloud cover (higher numbers are cloudier/darker)
        # 0 = Clear, 1 = Cirrus, 2 = Scattered, 3 = Broken, 4 = Overcast
        client.sendDREF("sim/weather/cloud_type[0]", CC)

        # Reset to the desired starting position
        client.sendDREF("sim/time/sim_speed", simSpeed)
        xpc3_helper.reset(client, cteInit = startCTE, heInit = startHE, dtpInit = startDTP)
        xpc3_helper.sendBrake(client, 0)

        time.sleep(1)  # 1 second to get terminal window out of the way
        client.pauseSim(False)

        dtp = startDTP
        startTime = client.getDREF("sim/time/zulu_time_sec")[0]
        endTime = startTime

        if logControl: 
            run_group = f.create_group(f'run_{i}')

        i = 0
        downImgTraj = []
        upImgTraj = []
        cteTraj = []
        dtpTraj = []
        rudderTraj = []


        while dtp < endDTP:
            # Deal with speed
            speed = xpc3_helper.getSpeed(client)
            throttle = 0.1
            if speed > 5:
                throttle = 0.0
            elif speed < 3:
                throttle = 0.2

            cteP, heP, downImg, upImg = getState(client)
            rudder = getControl(client, cteP, heP)
            client.sendCTRL([0, rudder, rudder, throttle])

            # Wait for next timestep
            while endTime - startTime < 1:
                endTime = client.getDREF("sim/time/zulu_time_sec")[0]
                time.sleep(0.001)

            # Set things for next round
            startTime = client.getDREF("sim/time/zulu_time_sec")[0]
            endTime = startTime
            cte, dtp, _ = xpc3_helper.getHomeState(client)
            time.sleep(0.001)

            cteTraj.append(cte)
            dtpTraj.append(dtp)
            upImgTraj.append(upImg)
            downImgTraj.append(downImg)
            rudderTraj.append(rudder)

            i += 1

        
        run_group.create_dataset('cte', data=cteTraj)
        run_group.create_dataset('dtp', data=dtpTraj)
        run_group.create_dataset('upImg', data=upImgTraj)
        run_group.create_dataset('downImg', data=downImgTraj)
        run_group.create_dataset('rudder', data=rudderTraj)



    client.pauseSim(True)

def main(): 

    with xpc3.XPlaneConnect() as client:

        startCTE = 5.0
        startHE  = 10.0
        startDTP = 322.0
        endDTP = 422.0

        run_sim(client, startCTE, startHE, startDTP, endDTP, 38005)


if __name__ == "__main__":
    main()