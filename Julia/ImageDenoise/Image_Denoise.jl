using Images
using LinearAlgebra
using BenchmarkTools
using DelimitedFiles 
using FileIO
using ColorTypes
using Colors

# Function to open an image in the same directory as the script
function open_image(filename)
    script_path = abspath(dirname(@__FILE__))
    image_path = joinpath(script_path, filename)
    return load(image_path)
end

function scale_gray_to_0_255(image::AbstractMatrix{<:Gray{N0f8}})
    # Convert the image to UInt8 directly
    scaled_image = reinterpret(UInt8, clamp01.(image))

    return scaled_image
end

function normalize_image(img::AbstractMatrix{<:Unsigned})
    # Convert the input image to Float64 and scale to [0, 1]
    normalized_img_array = Float64.(img) / 255.0

    return normalized_img_array
end

function get_image_from_directory(directory::AbstractString, image_name::AbstractString)
    # Get the directory path of the current script
    script_dir = dirname(@__FILE__)

    # Combine the script directory with the provided directory name
    directory_path = joinpath(script_dir, directory)

    # Check if the specified directory exists
    if isdir(directory_path)
        # Combine the directory path with the image name
        image_path = joinpath(directory_path, image_name)

        # Check if the image file exists
        if isfile(image_path)
            # Load and return the image
            return load(image_path)
        else
            println("Image file '$image_name' does not exist in directory '$directory'.")
        end
    else
        println("Directory '$directory' does not exist in the current path.")
    end

    return nothing  # Return nothing if the image does not exist
end

# Function to save an image as a PNG file
function save_image_as_png(output_image, filename::AbstractString)
    # Get the directory path of the current script
    script_dir = abspath(dirname(@__FILE__))

    # Combine the script directory with the provided filename
    file_path = joinpath(script_dir, filename)

    # Save the image as a PNG file in the same directory as the script
    save(file_path, output_image)
end

# Function to write a matrix to a file in the same folder as the script
function write_matrix_to_file(file_name::AbstractString, matrix::AbstractMatrix{T}) where T
    try
        # Get the directory path of the current script
        script_dir = dirname(@__FILE__)

        # Combine the script directory with the provided file name
        file_path = joinpath(script_dir, file_name)
        
        # Check if the directory exists, create it if it doesn't
        if !isdir(script_dir)
            mkdir(script_dir)
        end

        # Open the file in write mode
        open(file_path, "w") do file
            # If the file exists and is not empty, clear its content
            if isfile(file_path) && filesize(file_path) > 0
                truncate(file, 0)
            end

            # Write the new content to the file
            for row in eachrow(matrix)
                println(file, join(row, ", "))
            end
        end

        println("Content written to $file_path")
    catch e
        println("An error occurred: $e")
    end
end

# Function to read a text file with integer values in [0-255] scale and transform them into an image
function read_text_file_to_image(txt_file::AbstractString, output_image::AbstractString)
    # Get the directory path of the current script
    script_dir = dirname(@__FILE__)

    # Combine the script directory with the provided file names
    txt_file_path = joinpath(script_dir, txt_file)
    output_image_path = joinpath(script_dir, "$output_image.png")  # Specify PNG format here

    # Read the matrix from the text file as integers
    matrix = readdlm(txt_file_path, ',', Int)

    # Convert the matrix to Gray{N0f8} type with values in [0, 1] scale
    matrix = Gray.(matrix ./ 255.0)

    # Create an image from the matrix
    img = Gray.(matrix)

    # Save the image in the same directory as the script with the specified filename
    save(output_image_path, img)

    println("Image saved to $output_image_path")
end

function get_file_length(file_name::AbstractString)
    try
        # Get the directory path of the current script
        script_dir = dirname(@__FILE__)
        
        # Combine the script directory with the provided file name
        file_path = joinpath(script_dir, file_name)

        # Read the content from the file and split it into rows
        lines = readlines(file_path)

        # Calculate matrix dimensions (rows x columns)
        rows = length(lines)
        if rows > 0
            cols = length(split(strip(lines[1]), ','))
            println("Matrix dimensions: $rows rows x $cols columns")
        else
            println("Matrix is empty.")
        end

    catch e
        println("An error occurred: $e")
    end
end

function pinvFast(A)
    n = size(A, 2)
    m = size(A, 1)

    if m >= n
        M1 = A' * A
        y_1, v_1 = eigen(M1)
        consta = n * maximum(y_1) * eps(Float64)
        y_2 = y_1 .> consta
        r_A = sum(y_2)
        y_3 = abs.(y_1) .* y_2
        s_1, orden = sort(sqrt.(y_3), rev=true), sortperm(sqrt.(y_3), rev=true)
        v_2 = v_1[:, orden]
        v = v_2[:, 1:r_A]
        s = Diagonal(s_1[1:r_A])
        u = A * v / s
    else
        M1 = A * A'
        y_1, u_1 = eigen(M1)
        consta = m * maximum(y_1) * eps(Float64)
        y_2 = y_1 .> consta
        r_A = sum(y_2)
        y_3 = abs.(y_1) .* y_2
        s_1, orden = sort(sqrt.(y_3), rev=true), sortperm(sqrt.(y_3), rev=true)
        u_2 = u_1[:, orden]
        u = u_2[:, 1:r_A]
        s = Diagonal(s_1[1:r_A])
        v = A' * u / s
    end

    Ap = v * inv(s) * u'

    return Ap
end

function ImageDenoise()
    # Numerical Experiment 3

    # Reference: Soto-Quiros, P. (2022), A fast method to estimate the Moore-Penrose
    #            inverse for well-determined numerical rank matrices based on the
    #            Tikhonov regularization. (Submitted paper)

    numImg = 1110  # Number of images in folders "image_free_noise" and "image_with_noise"

    # Load training images
    X = zeros(Float64, 128^2, numImg)  # Images Free-Noisy
    Y = zeros(Float64, 128^2, numImg)  # Image with Noise

    for k in 1:numImg
        image_path = "image_free_noise"
        image_name = "coast ($k).jpg"
        X1 = get_image_from_directory(image_path, image_name)
        X2 = normalize_image(scale_gray_to_0_255(X1))
        X[:, k] = X2[:]

        image_path = "image_with_noise"
        image_name = "coast ($k).jpg"
        Y1 = get_image_from_directory(image_path, image_name)
        Y2 = normalize_image(scale_gray_to_0_255(Y1))
        Y[:, k] = Y2[:]
    end

    # Check Matrix T is rank-deficient
    T = Y' * Y
    m, n = size(T)
    println("Dimension of matrix T = $m x $n")
    r = rank(T)
    println("Matrix T is rank-deficient because rank(T) = $r")

    println("Dimensions of X: $(size(X))")

    # Compute Filter F1 (using proposed method) and F2 (using pinv)
    Yp1 = pinvFast(Y)
    t1 = @timed begin
        F1 = X * Yp1
    end

    println("Dimensions of Yp1: $(size(Yp1))")
    println("Execution time to compute Moore-Penrose of Y using proposed_method = $(t1[2]) seconds")
    println("Dimensions of F1: $(size(F1))")

    println("Dimensions of Y: $(size(Y))")

    Yp2 = pinv(Y)
    t2 = @timed begin
        F2 = X * Yp2
    end

    println("Dimensions of Yp2: $(size(Yp2))")
    println("Execution time to compute Moore-Penrose of Y using pinv = $(t2[2]) seconds")
    println("Dimensions of F2: $(size(F2))")

    # Perform analysis of proposed_method vs command pinv
    speedup = t2[2] / t1[2]
    per_dif = 100 * (t2[2] - t1[2]) / t2[2]
    println("Speedup to compute Moore-Penrose of Y using proposed_method = $speedup seconds")
    println("proposed_method is $per_dif% faster than command pinv")

    test_image_path = "test_images"

    # test_image_name = "test_image (1).jpg"
    test_image_name = "test_image (2).jpg"
    # test_image_name = "test_image (3).jpg"
    # test_image_name = "test_image (4).jpg"

    A1 = get_image_from_directory(test_image_path, test_image_name)
    Xt = normalize_image(scale_gray_to_0_255(A1))
    println("Dimensions of Xt: $(size(Xt))")
    
    # Add noise to the image
    noise = 0.1 * randn(size(Xt))
    Yt = Xt + noise

    # Reshape the noisy image to a 1D array
    yt_v = Yt[:]
    println("Dimensions of yt_v: $(size(yt_v))")
    
    # Perform the denoising using the proposed_method and pinv
    xt_v_pm = F1' * yt_v
    println("Dimensions of xt_v_pm: $(size(xt_v_pm))")
    Xt_est_pm = reinterpret(UInt8, clamp01.(reshape(xt_v_pm, size(Xt))))
    println("Dimensions of Xt_est_pm: $(size(Xt_est_pm))")
    #write_matrix_to_file("proposed_method_out_image.txt", Xt_est_pm)
    #get_file_length("proposed_method_out_image.txt")

    xt_v_pinv = F2' * yt_v
    println("Dimensions of xt_v_pm: $(size(xt_v_pinv))")
    Xt_est_pinv = reinterpret(UInt8, clamp01.(reshape(xt_v_pinv, size(Xt))))
    println("Dimensions of Xt_est_pinv: $(size(Xt_est_pinv))")
    #write_matrix_to_file("pinv_out_image.txt", Xt_est_pinv)
    #get_file_length("pinv_out_image.txt")
end

ImageDenoise()
