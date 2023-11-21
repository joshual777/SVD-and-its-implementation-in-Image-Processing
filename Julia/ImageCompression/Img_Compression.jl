using FilePaths
using Images
using LinearAlgebra
using BenchmarkTools
using DelimitedFiles 
using FileIO
using ColorTypes

# Function to open an image in the same directory as the script
function open_image(filename)
    script_path = abspath(dirname(@__FILE__))
    image_path = joinpath(script_path, filename)
    return load(image_path)
end

# Function to get the width and height of an image
function get_image_dimensions(image)
    return size(image)
end

function normalize_image(img::AbstractMatrix{<:Unsigned})
    # Convert the input image to Float64 and scale to [0, 1]
    normalized_img_array = Float64.(img) / 255.0

    return normalized_img_array
end

function scale_gray_to_0_255(image::AbstractMatrix{<:Gray{N0f8}})
    # Convert the image to UInt8 directly
    scaled_image = reinterpret(UInt8, image)
    return scaled_image
end

# Function to store an image as a PNG file
function store_image_as_png(image, filename)
    script_path = abspath(dirname(@__FILE__))
    output_path = joinpath(script_path, filename)
    
    try
        # Save the image as a PNG file
        save(output_path, image, format="jpeg")
        println("Image saved to $output_path")
    catch e
        println("An error occurred while saving the image: $e")
    end
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

# Function to read a matrix from a file in the same folder as the script
function read_matrix_from_file(file_name::AbstractString)
    try
        # Get the directory path of the current script
        script_dir = dirname(@__FILE__)

        # Combine the script directory with the provided file name
        file_path = joinpath(script_dir, file_name)

        # Read the matrix from the file using readdlm from DelimitedFiles
        return DelimitedFiles.readdlm(file_path, ',', Float64)
    catch e
        # println("An error occurred: $e")
        return nothing
    end
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

function svdJulia(A)
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

    return u, s, v
end


# Function to generate an image from a matrix in a text file
function generate_image_from_txt(txt_file::AbstractString, output_image::AbstractString)
    # Get the directory path of the current script
    script_path = abspath(dirname(@__FILE__))

    # Combine the script directory with the provided input and output file names
    input_txt_path = joinpath(script_path, txt_file)
    output_image_path = joinpath(script_path, output_image)

    # Read the matrix from the text file in the same directory
    matrix = readdlm(input_txt_path, ',', Float64)

    # Convert the matrix to an image
    img = Gray.(matrix)  # Assuming a grayscale image, change to RGB if needed

    # Save the image as a JPEG file in the same directory as the script
    save(output_image_path, img)
    
    println("Image saved to $output_image_path")
end

function transform_to_0_255_scale(input_filename::AbstractString, output_filename::AbstractString)
    # Get the directory path of the current script
    script_dir = dirname(@__FILE__)
    
    # Combine the script directory with the provided file names
    input_path = joinpath(script_dir, input_filename)
    output_path = joinpath(script_dir, output_filename)
    
    try
        # Read the matrix from the input text file
        matrix = readdlm(input_path, ',', Float64)

        # Scale the values from [0, 1] to [0, 255]
        scaled_matrix = round.(Int, matrix .* 255)

        # Write the scaled matrix to the output text file
        writedlm(output_path, scaled_matrix, ',')

        println("Scaled matrix saved to $output_path")
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

# Main code
function ImageCompression()

    # Open the image
    I1 = open_image("img1.jpg")

    I2 = scale_gray_to_0_255(I1)

    # Normalize and convert to Matrix{RGB}
    I = normalize_image(I2)

    # Extract the pixel data and convert to a suitable format for writing to a text file
    pixel_data = channelview(I)

    # Save the pixel data to a text file
    write_matrix_to_file("originalImage.txt", pixel_data)

    # Create compressed data
    r = 100
    U, S, V = svdJulia(pixel_data)
    Ur = U[:, 1:r]
    Sr = S[1:r, 1:r]
    Vr = V[:, 1:r]
    D = Ur * Sr
    C = Vr'
    Ic = D * C

    # Save the normalized image as a PNG file
    # store_image_as_png(Ic, "compressed_image.jpg")

    # Save matrices D and C to files
    write_matrix_to_file("C.txt", C)
    write_matrix_to_file("D.txt", D)

    # Load matrices D and C
    D1 = read_matrix_from_file("D.txt")
    C1 = read_matrix_from_file("C.txt")
    
    if D1 === nothing || C1 === nothing
        println("Error: Failed to load matrices D and C.")
        return
    end

    Ic1 = (D1 * C1) 

    write_matrix_to_file("Ic1.txt", Ic1)
   
    transform_to_0_255_scale("Ic1.txt", "O2.txt")

    read_text_file_to_image("O2.txt", "compressed_image_100")
end

ImageCompression()

