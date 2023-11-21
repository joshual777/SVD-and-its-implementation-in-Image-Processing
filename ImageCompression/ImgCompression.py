
"""
The next code aims to perform image compression using the proposed
Singular Value Decomposition (SVD) method
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Upload an image
def UploadImg(img_name):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    image_filename = img_name
    image_path = os.path.join(script_directory, image_filename)
    img = Image.open(image_path)
    return img

# Get image dimensions, width, and height
def GetImageSize(img):
    width, height = img.size
    print("Image width:", width)
    print("Image height:", height)

# Normalize image from 0-255 to 0 to 1
def NormalizedImg(img):
    img_array = np.array(img)
    normalized_img_array = img_array / 255.0
    return normalized_img_array

# Store Image in the current directory
def StoreImg(normalized_img_array, file_name):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    normalized_image_path = os.path.join(script_directory, file_name)
    normalized_img_pil = Image.fromarray((normalized_img_array * 255).astype(np.uint8))
    normalized_img_pil.save(normalized_image_path)

# Function to write content to a file in the same folder as the script
def write_to_file(file_name, content):
    try:
        # Get the directory path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Combine the script directory with the provided file name
        file_path = os.path.join(script_dir, file_name)

        # Open the file in write mode
        with open(file_path, 'w') as file:
            # If the file exists and is not empty, clear its content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                file.truncate(0)
            
            # Write the new content to the file
            file.write(content)

        #print(f"Content written to {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def get_file_length(file_name):
    try:
        # Get the directory path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Combine the script directory with the provided file name
        file_path = os.path.join(script_dir, file_name)

        # Read the content from the file and split it into rows
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Calculate matrix dimensions (rows x columns)
        rows = len(lines)
        if rows > 0:
            cols = len(lines[0].strip().split(','))
            print(f"Matrix dimensions: {rows} rows x {cols} columns")
        else:
            print("Matrix is empty.")

    except FileNotFoundError:
        print(f"File not found: {file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")



# Load matrices A and B
def LoadMatrices(file_name):
    try:
        # Get the directory path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Combine the script directory with the provided file name
        file_path = os.path.join(script_dir, file_name)

        # Read matrices from the file
        matrices = np.loadtxt(file_path)
        return matrices
    except Exception as e:
        #print(f"An error occurred: {str(e)}")
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

def ImageCompression():

    # Upload the original image
    img = UploadImg('img1.jpg')
    
    # Normalize the image
    normalized_img = NormalizedImg(img)
    
    # Display the original image
    plt.subplot(1, 3, 1)
    plt.imshow(normalized_img)
    plt.title('Original Image')
    
    # Store the normalized image
    StoreImg(normalized_img, 'normalized_image.jpg')

    # Write the normalized image to a text file
    normalized_image_txt = '\n'.join([','.join(map(str, row)) for row in normalized_img])
    write_to_file('originalImage.txt', normalized_image_txt)

    print(get_file_length('originalImage.txt'))

    # Perform image compression using SVD
    r = 10
    u, s, v = svdPython(normalized_img)
    Ur = u[:, :r]
    Sr = s[:r, :r]
    Vr = v[:, :r]
    D = Ur @ Sr
    C = Vr.T
    Ic = D @ C

    # Display the compressed image
    plt.subplot(1, 3, 2)
    plt.imshow(Ic)
    plt.title('Compressed Image')
    
    # Store matrices D and C
    write_to_file('D.txt', '\n'.join([','.join(map(str, row)) for row in D]))
    write_to_file('C.txt', '\n'.join([','.join(map(str, row)) for row in C]))

    print("D")
    print(get_file_length('D.txt'))
    print("C")
    print(get_file_length('C.txt'))
    
    # Load matrices D and C
    D1 = LoadMatrices("D.txt")
    C1 = LoadMatrices("C.txt")

    if D1 is not None and C1 is not None:
        Ic1 = D1 @ C1

        # Display the compressed image using loaded matrices D and C
        plt.subplot(1, 3, 3)
        plt.imshow(Ic1)
        plt.title('Compressed Image (Loading matrices D and C)')

    plt.show()
    
ImageCompression()

