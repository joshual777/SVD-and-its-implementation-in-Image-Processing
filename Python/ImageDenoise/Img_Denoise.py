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

# The provided Python function for Moore-Penrose pseudo-inverse
def pinvFast(A):
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

    Ap = np.dot(v, np.linalg.inv(s)).dot(u.T)
    return Ap

def ImageDenoise():
    # Numerical Experiment 3

    # Reference: Soto-Quiros, P. (2022), A fast method to estimate the Moore-Penrose
    #            inverse for well-determined numerical rank matrices based on the
    #            Tikhonov regularization. (Submitted paper)

    numImg = 1110  # Number of images in folders "image_free_noise" and "image_with_noise"

    # Load training images
    X = np.zeros((128 ** 2, numImg))  # Images Free-Noisy
    Y = np.zeros((128 ** 2, numImg))  # Image with Noise

    for k in range(1, numImg + 1):
        image_path = 'image_free_noise'
        image_name = f'coast ({k}).jpg'
        X1 = load_image_from_directory(image_path, image_name)  # Extract image name
        X2 = NormalizedImg(X1)
        X[:, k - 1] = X2.flatten()

        image_path = 'image_with_noise'
        image_name = f'coast ({k}).jpg'
        Y1 = load_image_from_directory(image_path, image_name)  # Extract image name
        Y2 = NormalizedImg(Y1)
        Y[:, k - 1] = Y2.flatten()

    numS = np.random.randint(1, numImg + 1, 6)

    # Show random images (free-noise)
    for k in range(len(numS)):
        plt.subplot(1, len(numS), k + 1)
        plt.imshow(X[:, numS[k] - 1].reshape((128, 128)), cmap='gray')
    plt.show()

    # Show random images (with noise)
    for k in range(len(numS)):
        plt.subplot(1, len(numS), k + 1)
        plt.imshow(Y[:, numS[k] - 1].reshape((128, 128)), cmap='gray')
    plt.show()

    # Check Matrix T is rank-deficient
    T = np.dot(Y.T, Y)
    m, n = T.shape
    #print(f'Dimension of matrix T = {m} x {n}')
    r = np.linalg.matrix_rank(T)
    #print(f'Matrix T is rank-deficient because rank(T) = {r}')

    #print(f'Dimensions of X: {X.shape}')

    # Compute Filter F1 (using proposed method) and F2 (using pinv)
    eps = np.finfo(float).eps
    Yp1 = pinvFast(Y)
    t1 = time.time()
    F1 = np.dot(X, Yp1)
    print(f'Execution time to compute Moore-Penrose of Y using proposed_method = {t1:.8f} seconds')
    #print(f'Dimensions of Yp1: {Yp1.shape}')
    #print(f'Dimensions of F1: {F1.shape}')

    #print(f'Dimensions of Y: {Y.shape}')

    Yp2 = np.linalg.pinv(Y)
    t2 = time.time()
    print(f'Execution time to compute Moore-Penrose of Y using pinv = {t2:.8f} seconds')
    #print(f'Dimensions of Yp2: {Yp2.shape}')

    F2 = np.dot(X, Yp2)
    #print(f'Dimensions of F2: {F2.shape}')

    # Perform analysis of proposed_method vs command pinv
    speedup = t2 / t1
    per_dif = 100 * (t2 - t1) / t2
    print(f'Speedup to compute Moore-Penrose of Y using proposed_method = {speedup:.8f} seconds')
    print(f'proposed_method is {per_dif:.8f}% faster than command pinv')

    # Error
    error = np.linalg.norm(Yp1 - Yp2, "fro") ** 2
    print(f'Error to compute Moore-Penrose of Y using proposed_method = {error}')

    # Clean noisy images - There are four test images
    test_image_path = 'test_images'
    
    # test_image_name = 'test_image (1).jpg'
    test_image_name = 'test_image (2).jpg'
    # test_image_name = 'test_image (3).jpg'
    # test_image_name = 'test_image (4).jpg'

    A1 = load_image_from_directory(test_image_path, test_image_name)
    Xt = NormalizedImg(A1)
    plt.imshow(Xt, cmap='gray')
    plt.title('Source Image')
    plt.show()

    Yt = Xt + 0.1 * np.random.randn(*Xt.shape)
    plt.imshow(Yt, cmap='gray')
    plt.title('Noisy Image')
    plt.show()
    yt_v = Yt.flatten()
    #print(f'Dimensions of yt_v: {yt_v.shape}')

    # Perform the denoising using proposed_method and pinv
    xt_v_pm = np.dot(F1, yt_v)
    #print(f'Dimensions of F2: {xt_v_pm.shape}')
    Xt_est_pm = np.uint8(xt_v_pm.reshape(Xt.shape) * 255)
    #print(f'Dimensions of F2: {Xt_est_pm.shape}')

    xt_v_pinv = np.dot(F2, yt_v)
    #print(f'Dimensions of F2: {xt_v_pinv.shape}')
    Xt_est_pinv = np.uint8(xt_v_pinv.reshape(Xt.shape) * 255)
    #print(f'Dimensions of F2: {Xt_est_pinv.shape}')

    # Display the estimated images and calculate errors
    plt.figure()
    plt.imshow(Xt_est_pm, cmap='gray')
    plt.title('Estimate Image with proposed_method')

    error_estimation_pm = np.linalg.norm(Xt - Xt_est_pm, 'fro') / np.linalg.norm(Xt, 'fro')
    print(f'Error estimation of proposed_method: {error_estimation_pm:.8f}')
    #print(f'Dimensions of Xt_est_pm: {Xt_est_pm.shape}')

    plt.figure()
    plt.imshow(Xt_est_pinv, cmap='gray')
    plt.title('Estimate Image with pinv')

    error_estimation_pinv = np.linalg.norm(Xt - Xt_est_pinv, 'fro') / np.linalg.norm(Xt, 'fro')
    print(f'Error estimation of pinv: {error_estimation_pinv:.8f}')
    #print(f'Dimensions of Xt_est_pinv: {Xt_est_pinv.shape}')

    # Show the plots
    plt.show()

ImageDenoise()
