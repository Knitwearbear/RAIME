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

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import keras

from keys import Keys



deadThreshold=30
fullHealthThreshold=280
resolution=np.array([150,200])
actionSize=10 #up, down, left, right, their sprint variants, nothing, shoot
actionThreshold=0.5
#list of available actions for Raime
UP=0
DOWN=1
LEFT=2
RIGHT=3

UP_SP=4
DOWN_SP=5
LEFT_SP=6
RIGHT_SP=7

NOTHING=8

SHOOT=9
#############end of action list


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
    healthRead=np.count_nonzero(output[:,:,0]) #371 is the highest I've seen it go but just in case
    #print(healthRead)
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
        
class GTAMovement:
    #0
    def moveUp(curMove):
        if curMove==UP:
            return
        ReleaseAllKeys()
        PressKey(W)
    #1    
    def moveDown(curMove):
        if curMove==DOWN:
            return
        ReleaseAllKeys()
        PressKey(S)
    #2
    def moveLeft(curMove):
        if curMove==LEFT:
            return
        ReleaseAllKeys()
        PressKey(A)
    #3
    def moveRight(curMove):
        if curMove==RIGHT:
            return
        ReleaseAllKeys()
        PressKey(D)
    #4
    def moveUpSprint(curMove):
        if curMove==UP_SP:
            return
        ReleaseAllKeys()
        PressKey(W)
        PressKey(SHIFT)
    #5
    def moveDownSprint(curMove):
        if curMove==DOWN_SP:
            return
        ReleaseAllKeys()
        PressKey(S)
        PressKey(SHIFT)
    #6
    def moveLeftSprint(curMove):
        if curMove==LEFT_SP:
            return
        ReleaseAllKeys()
        PressKey(A)
        PressKey(SHIFT)
    #7
    def moveRightSprint(curMove):
        if curMove==RIGHT_SP:
            return
        ReleaseAllKeys()
        PressKey(D)
        PressKey(SHIFT)
    #8
    def doNothing(curMove):
        if curMove==NOTHING:
            return
        ReleaseAllKeys()
    #9
    def shoot():
        keys=Keys()
        keys.directMouse(buttons=keys.mouse_lb_press)
        

    
        
        
class DQN:
    def __init__(self):
        self.attentionSpan=1000
        self.memory=deque(maxlen=self.attentionSpan)
        
        self.gamma=0.85
        self.epsilon=1.0
        self.epsilonMin=0.01
        self.epsilonDecay=0.995
        self.learningRate=0.001
        self.tau=0.125
        self.dreamDuration=60
        
        self.model=self.birth()
        self.targetModel=self.birth()

    def birth(self):
        convFilters_Layer1=96
        kernelSize_Layer1=11
        
        maxPoolSize_Layer2=3
        
        convFilters_Layer3=256
        kernelSize_Layer3=5
        
        maxPoolSize_Layer4=2
        
        denseSize_Layer5=4096
        denseSize_Layer6=4096
        denseSize_Output=actionSize
        
        dropoutProbability=0.5
        
        model=Sequential()
        model.add(Conv2D(convFilters_Layer1, kernel_size=(kernelSize_Layer1, kernelSize_Layer1),strides=4,
                     activation='relu',
                     input_shape= (resolution[0], resolution[1], 1)))
        model.add(MaxPooling2D(pool_size=(maxPoolSize_Layer2, maxPoolSize_Layer2), strides=4))
        model.add(Conv2D(convFilters_Layer3, (kernelSize_Layer3, kernelSize_Layer3), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(maxPoolSize_Layer4, maxPoolSize_Layer4), strides=4))
        #model.add(Conv2D(312, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(denseSize_Layer5, activation='tanh'))
        model.add(Dropout(dropoutProbability))
        model.add(Dense(denseSize_Layer6, activation='tanh'))
        model.add(Dropout(dropoutProbability))
        model.add(Dense(denseSize_Output, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=self.learningRate),
                  metrics=['accuracy'])
        return model
    
    def remember(self,state,action,reward,newState,done):
        self.memory.append([state,action,reward,newState,done])
    
    def dream(self):
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
                target[0][action]=reward + (QFuture*self.gamma)
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
            return np.argmax(np.random.rand(1,actionSize))
        else:
            return np.argmax(self.model.predict(state))
    
    def mindUpload(self, fn):
        self.model.save(fn)
        print("Raime was uploaded at: "+time.strftime("%Y-%m-%d %H:%M:%S"))
        
def mapDecisionToOutput(decision, move):
    keys=Keys()
    keys.directMouse(buttons=keys.mouse_lb_release)
    if decision == UP:
        GTAMovement.moveUp(move)
    elif decision == DOWN:
        GTAMovement.moveDown(move)
    elif decision == LEFT:
        GTAMovement.moveLeft(move)
    elif decision == RIGHT:
        GTAMovement.moveRight(move)
        
    elif decision == UP_SP:
        GTAMovement.moveUpSprint(move)
    elif decision == DOWN_SP:
        GTAMovement.moveDownSprint(move)
    elif decision == LEFT_SP:
        GTAMovement.moveLeftSprint(move)
    elif decision == RIGHT_SP:
        GTAMovement.moveRightSprint(move)
        
    elif decision == NOTHING:
        GTAMovement.doNothing(move)
        
    elif decision == SHOOT:
        GTAMovement.shoot()
    

        
def testShoot():
    time.sleep(3)
#    GTAMovement.shoot()
    keys=Keys()
    keys.directMouse(buttons=keys.mouse_lb_press)
    time.sleep(3)
    keys.directMouse(buttons=keys.mouse_lb_release)
    

        

#def main():
trials=100
trialLength=20000 #how many frames will be processed per trial. Comes to about 10 min right now.

timeBetweenDeaths=[]

print("Creating Raime. Please Wait.")
raime=DQN()
for i in list(range(5))[::-1]:
    print("Raime will rise in... " + str(i+1))
    time.sleep(1)
print("Raime has now been cursed with existence.")
paused=False
lastTime=time.time()

for trial in range(trials):
    screen=grab_screen(region=(0,28,799,627))
    health=detectHealth(screen)
    if health<fullHealthThreshold:
        while(health<fullHealthThreshold):
            screen=grab_screen(region=(0,28,799,627))
            health=detectHealth(screen)
        print("Raime enters the fray yet again.")
    
    screen=grab_screen(region=(0,28,799,627))
    screenSm=cv2.resize(screen, (resolution[1], resolution[0]))
    bwScreen=processImg(screenSm)
    currentState=bwScreen.reshape(1, resolution[0], resolution[1], 1)
    currentMovement=NOTHING
    timeOfBirth=time.time()
    
    for step in range(trialLength):
        if not paused:
            raimesDecision=raime.theTimeForActionIsNow(currentState)
            mapDecisionToOutput(raimesDecision, currentMovement)
            currentMovement=raimesDecision
            
            screen=grab_screen(region=(0,28,799,627))
            screenSm=cv2.resize(screen, (resolution[1], resolution[0]))
            reward=detectHealth(screen)
            
            #experimental
            done=False #will implement feature later
            if reward<deadThreshold:
                done=True
           
            bwScreen=processImg(screenSm)
            newState=bwScreen.reshape(1, resolution[0], resolution[1], 1)
            
            raime.remember(currentState, raimesDecision, reward, newState, done)
            
            raime.dream()
            raime.targetTrain()
          
            currentState=newState
            
            if done:
                ReleaseAllKeys()
                lifespan=time.time()-timeOfBirth
                timeBetweenDeaths.append(lifespan)
                print("Raime died.")
                plt.plot(timeBetweenDeaths)
                plt.ylabel('Time Survived (s)')
                plt.xlabel('Trial')
                plt.title('Survival Over Training')
                plt.show()
                break
            
            
    # show what raime sees        
#            cv2.imshow('window', bwScreen)
#            if cv2.waitKey(1) & 0xFF == ord('q'):
#                cv2.destroyAllWindows()
#                break
           
            if step%1000==0:
                raime.mindUpload("raime1-5.h5")
                
                
            #These 2 lines print the frames Raime can process per second. Totally optional
            print(str(1/(time.time()-lastTime)) + " fps" )
            lastTime=time.time()
        else:
            while(paused):
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
ReleaseAllKeys()
raime.mindUpload("raime1-5.h5")

#if __name__=="__main__":
#    main()






