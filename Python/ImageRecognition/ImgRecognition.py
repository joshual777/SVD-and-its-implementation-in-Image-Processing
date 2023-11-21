import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Normalize image from 0-255 to 0 to 1
def NormalizedImg(img):
    img_array = np.array(img)
    normalized_img_array = img_array / 255.0
    return normalized_img_array

def load_image_from_directory(directory_name, image_name):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_directory, directory_name, image_name)

    if os.path.exists(image_path):
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            print(f"Failed to open image '{image_name}' in directory '{directory_name}': {e}")
            return None
    else:
        print(f"Image '{image_name}' not found in directory '{directory_name}'.")
        return None


# The next section contains the implementation of the SVD algorithm 
def svdPython(A):

    m, n = A.shape         

    if m >= n:              
        M1 = A.T @ A        
        y_1, v_1 = np.linalg.eigh(M1)  
        const = n * np.max(y_1) * np.finfo(float).eps 
        y_2 = (y_1 > const) 
        r_A = np.sum(y_2)   
        y_3 = np.abs(y_1) * y_2 
        s_1, orden = np.sort(np.sqrt(y_3), axis=0)[::-1], np.argsort(np.sqrt(y_3), axis=0)[::-1]
        v_2 = v_1[:, orden] 
        v = v_2[:, :r_A]    
        s = np.diag(s_1[:r_A])  
        u = (1. / s_1[:r_A]) * (A @ v) 

    else:
        M1 = A @ A.T 
        y_1, u_1 = np.linalg.eigh(M1)   
        const = m * np.max(y_1) * np.finfo(float).eps  
        y_2 = (y_1 > const)  
        r_A = np.sum(y_2)    
        y_3 = np.abs(y_1) * y_2 
        s_1, orden = np.sort(np.sqrt(y_3), axis=0)[::-1], np.argsort(np.sqrt(y_3), axis=0)[::-1]
        u_2 = u_1[:, orden]  
        u = u_2[:, :r_A]     
        s = np.diag(s_1[:r_A])  
        v = (1. / s_1[:r_A]) * (A.T @ u) 
        
    return u, s, v


def ImageRecognition():

    # Load face images into matrix A
    numPerson = 40
    numImg = 9
    A = []

    for i in range(1, numPerson + 1):
        for j in range(2, numImg + 2):
            text = f'dataset_jpg/s{i}/{j}.jpg'
            Aux = np.array(load_image_from_directory("dataset_jpg", f's{i}/{j}.jpg'), dtype=np.float64)
            A.append(Aux.flatten())

    A = np.array(A).T
    A1 = A - np.mean(A, axis=1).reshape(-1, 1)
    tic = time.time()
    Ur, Sr, Vr = svdPython(A1)
    t1 = time.time() - tic
    X = np.dot(Ur.T, A1)


    # Identify new faces
    for numPersonNew in range(1, 41):
        text = f'dataset_jpg/s{numPersonNew}/1.jpg'
        newIm = np.array(load_image_from_directory("dataset_jpg", f's{numPersonNew}/1.jpg'), dtype=np.float64)
        f = newIm.flatten() - np.mean(A, axis=1)

        XnewImg = np.dot(Ur.T, f)

        """"""
        #print(X.shape)
        #print(len(X[0]))
        #print(XnewImg[0])
        #print(len(XnewImg))
        #print(XnewImg.shape)

        #m1 = np.zeros(X.shape[1])  # Assuming X.shape[1] is the number of columns in X

        # Use a for loop to calculate the differences for each column
        #for i in range(X.shape[1]):
        #    m1[i] += np.linalg.norm((X[:, i] - XnewImg ), axis=0 )
        #    print( m1[i])

        result = X - XnewImg[:, np.newaxis]
        
        errorsCoordinates = np.linalg.norm(result, axis=0)
        #print(errorsCoordinates)

        #print( m1)
        #m1 = X- XnewImg[:,None]
        #print(m1.shape)
        #errorsCoordinates = np.linalg.norm(m1, axis=0)
        idx = np.argmin(errorsCoordinates)
        #print(idx)
        k = idx
        print(k)
        plt.subplot(1, 2, 1)
        plt.imshow(newIm, cmap='gray')
        plt.title(f'New Face - Person #{numPersonNew}')

        plt.subplot(1, 2, 2)
        identIm = np.reshape(A[:, k], (112, 92))
        plt.imshow(identIm, cmap='gray')
        plt.title('Face Identified')
        plt.xlabel(f'Error = {errorsCoordinates[0]:.8f}')
        plt.pause(1)
    plt.show()


ImageRecognition()