using FilePaths
using Images
using LinearAlgebra
using BenchmarkTools
using DelimitedFiles 
using FileIO
using ColorTypes
using Colors

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
            println(image_path)
            return load(image_path)
        else
            println("Image file '$image_name' does not exist in directory '$directory'.")
        end
    else
        println("Directory '$directory' does not exist in the current path.")
    end

    return nothing  # Return nothing if the image does not exist
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

function ImageRecognition()
    numPerson = 40
    numImg = 9

    A = []

    for i in 1:numPerson
        for j in 2:numImg+1
            text = "dataset_jpg/s$i/"
            Aux = scale_gray_to_0_255(get_image_from_directory(text, "$j.jpg"))
            println("Dimensions of Aux: $(size(Aux))")
            A = hcat(A, Aux[:])
            println("Dimensions of A: $(size(A))")
        end
    end

    A1 = A .- mean(A, dims=2)
    Ur, Sr, Vr = svdJulia(A1)

    # Coordenadas 
    X = Ur' * A1

    # Nueva imagen
    for numPersonNew in 1:40
        text = "dataset_jpg/s$numPersonNew/1.jpg"
        newIm = scale_gray_to_0_255(get_image_from_directory(text))
        f = newIm[:] .- mean(A, dims=2)[:]

        XnewImg = Ur' * f

        erroresCordenadas = vecnorm(X .- XnewImg)

        minErroresCordenadas, idx = findmin(erroresCordenadas)

        k = idx[1]

        display(newIm, title = "New Face - Person # $numPersonNew")
        identIm = reinterpret(UInt8, clamp01(reshape(A[:, k], (112, 92))))
        display(identIm, title = "Face Identified")
        println("Error = $minErroresCordenadas")
    end
end

ImageRecognition()
