"""
@author: Knitwearbear, with the help of the internet
"""

import numpy as np
from grabscreen import grab_screen
import cv2
import random
import time
from directkeys import PressKey, ReleaseKey, ReleaseAllKeys, W,A,S,D, SHIFT
from getkeys import key_check

from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import keras



resolution=np.array([600,800])
actionSize=5
actionThreshold=0.5

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
    healthRead=np.count_nonzero(output[:,:,0])/380 #371 is the highest I've seen it go but just in case
    return healthRead

def raimeControl(action):
    if action[0]>actionThreshold:
        PressKey(W)
    else:
        ReleaseKey(W)
        
    if action[1]>actionThreshold:
        PressKey(A)
    else:
        ReleaseKey(A)
        
    if action[2]>actionThreshold:
        PressKey(S)
    else:
        ReleaseKey(S)
        
    if action[3]>actionThreshold:
        PressKey(D)
    else:
        ReleaseKey(D)
        
    if action[4]>actionThreshold:
        PressKey(SHIFT)
    else:
        ReleaseKey(SHIFT)
        
c
        
        
class DQN:
    def __init__(self):
        self.memory=deque(maxlen=2000)
        
        self.gamma=0.85
        self.epsilon=1.0
        self.epsilonMin=0.01
        self.epsilonDecay=0.995
        self.learningRate=0.005
        self.tau=0.125
        self.dreamDuration=32
        
        self.model=self.creation()
        self.targetModel=self.creation()

    def creation(self):
        convFilters_Layer1=50
        kernelSize_Layer1=10
        
        maxPoolSize_Layer2=4
        
        convFilters_Layer3=50
        kernelSize_Layer3=10
        
        maxPoolSize_Layer4=4
        
        denseSize_Layer5=256
        denseSize_Layer6=128
        denseSize_Output=actionSize
        
        model=Sequential()
        model.add(Conv2D(convFilters_Layer1, kernel_size=(kernelSize_Layer1, kernelSize_Layer1),
                     activation='relu',
                     input_shape= (resolution[0], resolution[1], 1)))
        model.add(MaxPooling2D(pool_size=(maxPoolSize_Layer2, maxPoolSize_Layer2)))
        model.add(Conv2D(convFilters_Layer3, (kernelSize_Layer3, kernelSize_Layer3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(maxPoolSize_Layer4, maxPoolSize_Layer4)))
        model.add(Flatten())
        model.add(Dense(denseSize_Layer5, activation='relu'))
        model.add(Dense(denseSize_Layer6, activation='relu'))
        model.add(Dense(denseSize_Output, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=self.learningRate),
                  metrics=['accuracy'])
        return model
    
    def remember(self,state,action,reward,newState,done):
        self.memory.append([state,action,reward,newState,done])
    
    def replay(self):
        if len(self.memory)<self.dreamDuration:
            return
        samples=random.sample(self.memory, self.dreamDuration)
        for sample in samples:
            state, action, reward, newState, done=sample
            target=self.targetModel.predict(state)
            if done:
                target[0][action]=reward
            else:
                QFuture=max(self.targetModel.predict(newState)[0])
                target[0][action]=reward
            self.model.fit(state,target, epochs=1, verbose=0)
            
            
    def targetTrain(self):
        weights=self.model.get_weights()
        targetWeights=self.targetModel.get_weights()
        for i in range(len(targetWeights)):
            targetWeights[i]=weights[i]*self.tau +targetWeights[i]*(1-self.tau)
        self.targetModel.set_weights(targetWeights)
        
    def theTimeForActionIsNow(self, state):
        self.epsilon*=self.epsilonDecay
        self.epsilon=max(self.epsilonMin, self.epsilon)
        if np.random.random()<self.epsilon:
            return np.random.rand(1,actionSize)
        else:
            return self.model.predict(state)
    
    def mindUpload(self, fn):
        self.model.save(fn)
        

#def main():
for i in list(range(5))[::-1]:
    print("Booting Raime... " + str(i+1))
    time.sleep(1)

paused=False
lastTime=time.time()
while(True):
    if not paused:
        screen=grab_screen(region=(0,28,799,627))
        health=detectHealth(screen)
        #print(health)
        screen=processImg(screen)
        currentState=screen.reshape(1, resolution[0], resolution[1], 1)

        #screen=cv2.resize(screen, (400, 300))
        
#        cv2.imshow('window', screen)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            cv2.destroyAllWindows()
#            break
        
        raimeControl(np.array([1,1,0,0,1]))
        
        #These 2 lines print the frames Raime can process per second. Totally optional
        print(str(1/(time.time()-lastTime)) + " fps" )
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

#if __name__=="__main__":
#    main()






