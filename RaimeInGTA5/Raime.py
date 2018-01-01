"""
@author: Knitwearbear
"""

import numpy as np
from grabscreen import grab_screen
import cv2
#import os
import time
from directkeys import PressKey, ReleaseKey, ReleaseAllKeys, W,A,S,D, SHIFT
from getkeys import key_check

def processImg(original):
    processed=cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    return processed

def detectHealth(image):
    lower=np.array([59,126,58]) #rgb value of healthbar 79,146,78
    upper=np.array([99,166,98])
    mask=cv2.inRange(image, lower, upper)
    output=cv2.bitwise_and(image, image, mask=mask)
    
    rect= np.array([[0,570], [0,760],[100,760], [100,570]], np.int32)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask, [rect], 255)
    output=cv2.bitwise_and(output,mask)
    return output

def raimeControl(action):
    if action[0]==1:
        PressKey(W)
    else:
        ReleaseKey(W)
        
    if action[1]==1:
        PressKey(A)
    else:
        ReleaseKey(A)
        
    if action[2]==1:
        PressKey(S)
    else:
        ReleaseKey(S)
        
    if action[3]==1:
        PressKey(D)
    else:
        ReleaseKey(D)
        
    if action[4]==1:
        PressKey(SHIFT)
    else:
        ReleaseKey(SHIFT)

    
for i in list(range(5))[::-1]:
    print("Booting Raime... " + str(i+1))
    time.sleep(1)

actionSize=5
paused=False
lastTime=time.time()
while(True):
    if not paused:
        screen=grab_screen(region=(0,28,799,627))
        filtered=detectHealth(screen)
        health=np.count_nonzero(filtered[:,:,0])
        print(health)
        screen=processImg(screen)
        #screen=cv2.resize(screen, (400, 300))
        cv2.imshow('window', screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        #PressKey(W)
        raimeControl(np.array([1,1,0,0,1]))
        #print(str(1/(time.time()-lastTime)) + " fps" )
        lastTime=time.time()
    keys=key_check()
    if 'T' in keys:
        if paused:
            paused = False
            for i in list(range(5))[::-1]:
                print("Unpausing... " + str(i+1))
                time.sleep(1)
            print('Unpaused.')
        else:
            ReleaseAllKeys()
            print ('Paused.')
            paused=True
            time.sleep(1)








