# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 15:47:17 2021

@author: Hadeer,Fazeela,Roaa
"""
import utils as p
import numpy as np
import os

def getAccuracy(actual,predictions):
    """
    function receives two single dimensional numpy arrays (predictions and
    actual) then returns the accuracy of the predictions compared with the
    actual labels.
    
    accuracy is out of 100 percent.
    
    Parameters:
    ----------
    predictions: 1-D numpy array with predictions (either 1, or -1)
    actual:      1-D numpy array with actual labels (either 1, or -1)
    """
    
    a=predictions==actual
    

    return len(predictions[a])/len(actual)*100



def calculate_recall(actuals, predictions):
    """
    This function calculates the RECALL for binary classification predictions.
    It receives two 1-D numpy arrays actuals and predictions and returns one number
    which is the recall. (range 0-1)
    
  
    Parameters:
    ----------
    actuals: 1-D numpy array with ACTUAL labels (either -1 = negative, or 1 = positive)
    predictions: 1-D numpy array with PREDICTED labels (either  = negative, or 1 = positive)
    
    """
    TP=0
    AP=len(actuals[actuals==1])
    for i,n in enumerate(predictions):
        if n==1:
            if n==actuals[i]:
                TP+=1
        
    return TP/AP

def calculate_precision(actuals, predictions):
    """

    This function calculates the PRECISION for a binary classification predictions.
    It receives two 1-D numpy arrays actuals and predictions. and returns one number
    which is the r. (range 0-1)
    
    Parameters:
    ----------
    actuals: 1-D numpy array with ACTUAL labels (either -1 = negative, or 1 = positive)
    predictions: 1-D numpy array with PREDICTED labels (either 1 = negative, or 1 = positive)
    
    """
    TP=0
    PredP=len(predictions[predictions==1])
    for i,n in enumerate(predictions):
        if n==1:
            if n==actuals[i]:
                TP+=1
        
    
    return TP/PredP

def calculate_f1(actuals, predictions):
    """
    This function calculates the F1 for a binary classifiers predictions.
    It receives two 1-D numpy arrays actuals and predictions. and returns one number
    which is the recall. (range 0-1)
    
    
    Parameters:
    ----------
    actuals: 1-D numpy array with ACTUAL labels (either 0 = negative, or 1 = positive)
    predictions: 1-D numpy array with PREDICTED labels (either 0 = negative, or 1 = positive)
    
    """
    P=calculate_precision(actuals,predictions)
    R=calculate_recall(actuals,predictions)
    f1=(2*P*R)/(P+R)
    
    return f1 

def macro_recall(classesRecall):
    """
    This function calculates the Macro recall for the multi classifier  
    
    
    Parameters:
    ----------
    classesRecall: 1D array of calculated recalls for all 13 classes
    """
    
    return np.average(classesRecall)
    
def macro_Precision(classesPercision):
    """
    This function calculates the Macro Percision for the multi classifier 
    
    
    Parameters:
    ----------
    classesPercision: 1D array of calculated Percision for all 13 classes
    """
    return np.average(classesPercision)

def macro_F1(classesF1):
    """
    This function calculates the Macro F1 for the multi classifier 
    
    
    Parameters:
    ----------
    classesF1: 1D array of calculated F1 for all 13 classes
    """
    
    return np.average(classesF1)


def get_files(directory):
    """
    This function is just to demonstrate how you can access all the files
    inside a directory.
    """
    files=[]
    for file in os.listdir(directory):
        f=directory+"/"+file
        files.append(f)
        
    return np.array(files)



   
def calculate_Feature(regions,fold):
    '''This Function Will calculate Min,MAx,Mean and STD For the given region
    Parameters:
        Regions: 
    
    '''
    start=(fold-1)*1000
    end=start+1000
    subregion=regions[start:end]
    MIN=np.min(subregion)
    MAX=np.max(subregion)
    MEAN=np.mean(subregion)
    STD=np.std(subregion)
    return np.array([MIN,MAX,MEAN,STD])


    
   
    
def Extract_All_Features(path):
    
    '''This function will extract the features from the given path of  an image
    and return it as 1D numpy array
    Parmeters:
        path--> Path for an image
        
    OUTPUT:
        1D array of extracted Features
    
    '''
    
    img=p.get_image_1d(path) # Get 1D image data
    #Extract Colors From 1d image
    red=img[::3]
    green=img[1::3]
    blue=img[2::3]
    
    No_reg=9
    feature_per_color=4*3 # 3 Colors(blue,red,green), 4 Calculations(Min,max,sd,mean)
    No_features=No_reg*feature_per_color #Calculate total Number of Features
    
    all_features=np.array([0.0]*No_features) 
    
    for i in range(1,No_reg+1):
        region_features=np.array([0.0]*feature_per_color)
        region_features[0:4]=calculate_Feature(red,i) # Calculate min,max,mean,std for Region i Color red
        region_features[4:8]=calculate_Feature(blue,i) # Calculate min,max,mean,std for Region i Color blue
        region_features[8:] =calculate_Feature(green,i) # Calculate min,max,mean,std for Region i Color green
        
        
        #Append red,blue,green together in one array
        start_index=(i-1)*feature_per_color
        for k in range(0,feature_per_color):
            all_features[start_index+k]=region_features[k]
            
    #Normaliz Features   
    norm = np.linalg.norm(all_features)
    Features= (all_features/norm)

    return Features
        
        


def imageToData(files):
 
    '''
    This Function will generate the dataset for a given array of images
    1- Extract Features
    2- Assign Label
    3- Generate Dataset
    '''

    images=np.array([Extract_All_Features(file) for file in files])
    for n,i in enumerate(files):
        if "apple" in i:
            images[n,images.shape[1]-1]=0
        elif "banana" in i:
            images[n,images.shape[1]-1]=1
        elif "cherry" in i:
            images[n,images.shape[1]-1]=2
        elif "date" in i:
            images[n,images.shape[1]-1]=3
        elif "ginger" in i:
            images[n,images.shape[1]-1]=4
        elif "grapefruit" in i:
            images[n,images.shape[1]-1]=5
        elif "lemon" in i:
            images[n,images.shape[1]-1]=6
        elif "lime" in i:
            images[n,images.shape[1]-1]=7
        elif "mango" in i:
            images[n,images.shape[1]-1]=8
        elif "orange" in i:
            images[n,images.shape[1]-1]=9
        elif "strawberry" in i:
            images[n,images.shape[1]-1]=10
        elif "tomato" in i:
            images[n,images.shape[1]-1]=11
        elif "watermelon" in i:
            images[n,images.shape[1]-1]=12
            
        
    
    return images

'''Get images'''   
path='data'
Files=get_files(path)
data=[]


for i in Files:
    images=get_files(i)
    data.append(images)

'''Split images to 70% Training and 30% Testing'''     
data=np.array(data)        
TrainingImages=[]
TestingImages=[]
for j in data:
    for a in range(0,int(np.floor(0.7*len(j)))):
        TrainingImages.append(j[a])
   
    for c in range(int(np.floor(0.7*len(j))),len(j)):
       TestingImages.append(j[c])

TrainingImages=np.array(TrainingImages)
TestingImages=np.array(TestingImages)

'''Generate training Dataset from Training Images'''
TrainingData=imageToData(TrainingImages)
 

       
def percep_learn(epoch,data):
    
    '''This Function Will Apply the perceptron Algorithm to the given dataset and generate
    a 1D array of Weights 
    
    1- Initialize Weight And Bias
    2- Initialize Learning rate
    3- for each data entry apply perceptron function and calculate output
    4- Calculate error 
    5- Update Weight and bias
    6- Repeat 3,5 for given no. of epochs
    '''
    eta=0.1
    
    w=np.array([0.81 for i in range(data.shape[1])])
    w[0]=0

    for j in range(epoch):
        for i in data:
            feature=i[0:data.shape[1]-1] 
            
            
            hx=np.dot(feature.T,w[1:])+w[0] #Calculate Output 
            
            cx=i[data.shape[1]-1] #Get Actual output [label]
            
            err=cx-hx #Calculate Error
     
            if err!=0:
                w[1:]=w[1:]+(eta*err*feature) #Update Weights
                w[0]=w[0]+(eta*err) #Update Bias
                        
    return w



def one_class_Data(data,fruit):
    Data=data.copy()
    for i in Data:
        #print(i[-1])
        if i[-1]==fruit:
            i[-1]=1
        else:
            i[-1]=-1
    
    return Data
    
    

def fruit_train(fruit,epochs):
    '''
    This Function Will Train the perceptron for an individual class [Fruit] with a given no of epochs
    and return the weights array for this fruit
    
    Parameters:
     fruit --> Fruit Label
     ephocs--> No of epochs when training Percpetron
     
     1- Split Training Data to fruit (i) and Not fruit
     2- Assign tmp labels 1 and -1
     3- Perform Oversampling
     4-train (Generate Weights)
     
    '''
    data=one_class_Data(TrainingData,fruit)
    class1=data[data[:,data.shape[1]-1]==1]
    class2=data[data[:,data.shape[1]-1]==-1]
   
    posClass=class1
    while len(posClass)<len(class2):
                     
         posClass=np.concatenate((posClass,class1[0:int(len(class1))]),axis=0)
 
          
    train=np.concatenate((posClass,class2),axis=0)
    np.random.shuffle(train)
    w=percep_learn(epochs,train)
     
    return w


def calculate_output(x,w):
    '''
    This function Will estimate the label for a given input [x] 
    Parameters:
        x--> Input (1D array of Features)
        w--> can be 1D array For a single fruit classification or No_classes X No_features Matrix for multi_class Classification
    
    OUTPUT:
        if w is 1D array then a will be single value
        if w is  No_classes X No_features Matrix then a will be 1D array (length=No_Classes)
    '''
    a = w[0] + np.dot(x.T,w[1:])  
 
    return a






def predict_Single_Class(x,fruit,w):    
      '''
      This function Will decide if x belongs to class fruit or not
      Parameters:
          x--> Path of img to classify
          fruit--> fruit class Label
    
       OUTPUT:
          The fuction will return 1 if x belongs to class fruit 
          The function will return -1 if x doesn't belong to class fruit
      '''
    
      img=Extract_All_Features(x)
      a=calculate_output(img[0:len(img)-1],w)
      if a>0:
           return 1
      else:
           return -1
       
     
            
              

    
'''Perform Training and save weights for single class classification and OVA Classification

for single class classification output will be caluclated using fruitWeights[Fruit] 1D array a=calculate_output(img,fruitWeights[Fruit])
for multiclass classification output will be caluclated using fruitWeights Matrix as a whole a=calculate_output(img,fruitWeights)
 '''    

fruitWeights=np.array([fruit_train(i,15) for i in range(13)]) # epochs=15 weights will be used for single class classification & multiclass classification

np.savetxt('fruitWeights.csv', fruitWeights, delimiter=',')
data=np.loadtxt('fruitWeights.csv', delimiter=',')
fruitWeights1=np.array([fruit_train(i,14) for i in range(13)]) #calculated to assist with multiclass classification
# f2 = open("fruitWeights1.txt", "w")

fruitWeights2=np.array([fruit_train(i,13) for i in range(13)]) #calculated to assist with multiclass classification
# f3 = open("fruitWeights2.txt", "w")




    
def OVA_vote(img):
        '''
        This function Will predict the label for a given img 3 times with different Weight Matrices
        and return the most repeated predection [if equal no. of predection: choose randomly]
        
        Parameters:
            img --> 1D array of features [input]
            
        OUTPUT:
            predicted class for the give input (img)
            
        '''
        
        # argmax() is used to get index of maximum value in the array returned from calculate_output
        
        clsf1=np.argmax(calculate_output(img[0:len(img)-1],fruitWeights.T)) #First Predection [0-12]
        clsf2=np.argmax(calculate_output(img[0:len(img)-1],fruitWeights1.T)) #Second Predection [0-12]
        clsf3=np.argmax(calculate_output(img[0:len(img)-1],fruitWeights2.T)) #Third Prediction [0-12]

        c=np.array([clsf1,clsf2,clsf3])
     
        counts = np.bincount(c) #Count Repetition 
        
        if max(counts)==1:
            return np.random.choice(c) #Choose random number if no repetition 
        else:
            return np.argmax(counts) #Return most repeated prediction
        # return clsf1


def predict_fruit(image):
    
    '''This Function will take the path for an image and return the correspodning classification for it
    
    Parameters:
        image --> image path 
        
    OUTPUT:
        an INT [0-12] to indicate the predicted class foe 
        
        
    
    '''
    img=Extract_All_Features(image)
    clsf=OVA_vote(img)
    
    return clsf
   


   




fruit=["Apple","Banana","Cherry","Dates","Ginger","Grapefruit","Lemon","Lime","Mango","Orange","Strawberry","Tomato","Watermelon"]


'''Perform Testing on Trianing Data Set (SINGLE CLASS)'''


predictions_tr=np.zeros(len(TrainingImages),)
actual_tr=np.zeros(len(TrainingImages),)
precision_tr=np.zeros(13,)
recall_tr=np.zeros(13,)
f1_tr=np.zeros(13,)

for i in range(13):
    
    fruitData=one_class_Data(TrainingData,i)
    
    for n,imgtr in enumerate(TrainingImages):
    
        result=predict_Single_Class(imgtr,i,fruitWeights[i])     
        predictions_tr[n]=result
        actual_tr[n]=fruitData[n,-1]
   

    accuracy_tr=getAccuracy(actual_tr,predictions_tr)
    precision_tr[i]=calculate_precision(actual_tr, predictions_tr)
    recall_tr[i]=calculate_recall(actual_tr,predictions_tr)
    f1_tr[i]=calculate_f1(actual_tr,predictions_tr)
    print(fruit[i])
    print("Trianing Data Accuracy: ", accuracy_tr)
    print("Trianing Data Precision: ", precision_tr[i])
    print("Trianing Data Recall ", recall_tr[i])
    print("Trianing Data f1: ", f1_tr[i])


'''Perform Testing on Testing Data Set (SINGLE CLASS)'''
TestingData=imageToData(TestingImages)

precision_tst=np.zeros(13,)
recall_tst=np.zeros(13,)
f1_tst=np.zeros(13,)

for i in range(13):
    predictions_tst=np.zeros(len(TestingImages),)
    actual_tst=np.zeros(len(TestingImages),)
    fruitData=one_class_Data(TestingData,i)
    
    for n,imgtst in enumerate(TestingImages):
    
        result=predict_Single_Class(imgtst,i,fruitWeights[i])     
        predictions_tst[n]=result
        actual_tst[n]=fruitData[n,-1]
        

    accuracy_tst=getAccuracy(actual_tst,predictions_tst)
    precision_tst[i]=calculate_precision(actual_tst, predictions_tst)
    recall_tst[i]=calculate_recall(actual_tst,predictions_tst)
    f1_tst[i]=calculate_f1(actual_tst,predictions_tst)
    print(fruit[i])
    print("Testing Data Accuracy: ", accuracy_tst)
    print("Testing Data Precision: ", precision_tst[i])
    print("Testing Data Recall ", recall_tst[i])
    print("Testing Data f1: ", f1_tst[i])
    
    
'''Training Data Set Performance (Multi CLASS)'''
MC_predictions_tr=np.zeros(len(TrainingData))
MC_actual_tr=np.zeros(len(TrainingData))
for n,MCimgtr in enumerate(TrainingImages):
    
        result=predict_fruit(MCimgtr)
        MC_predictions_tr[n]=result
        MC_actual_tr[n]=(TrainingData[n][-1])    
        
overallAccuracy_tr=getAccuracy(MC_actual_tr,MC_predictions_tr)
macroPrecision_tr=macro_Precision(precision_tr)
macroRecall_tr=macro_recall(recall_tr)
macroF1_tr=macro_F1(f1_tr)
print("Trianing Data overall Accuracy: ", overallAccuracy_tr)
print("Trianing Data Precision: ", macroPrecision_tr)
print("Trianing Data Recall ", macroRecall_tr)
print("Trianing Data f1: ", macroF1_tr)

'''Testing Data Set Performance (Multi CLASS)'''
MC_predictions_tst=np.zeros(len(TestingData))
MC_actual_tst=np.zeros(len(TestingData))
for n,MCimgtst in enumerate(TestingImages):
    
        result=predict_fruit(MCimgtst)
        MC_predictions_tst[n]=result
        MC_actual_tst[n]=(TestingData[n][-1]) 
        #p.show_image(p.get_image_3d(MCimgtst)) #uncomment this Line to Display images
      
overallAccuracy_tst=getAccuracy(MC_actual_tst,MC_predictions_tst)
macroPrecision_tst=macro_Precision(precision_tst)
macroRecall_tst=macro_recall(recall_tst)
macroF1_tst=macro_F1(f1_tst)
print("Testing Data Overrall Accuracy: ", overallAccuracy_tst)
print("Testing Data Macro Precision: ", macroPrecision_tst)
print("Testing Data Macro Recall ", macroRecall_tst)
print("Testing Data Macro f1: ", macroF1_tst)
    
    

  

        
    
    