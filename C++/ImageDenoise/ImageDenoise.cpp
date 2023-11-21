#include <fstream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <fstream>
#include <string>
#include <armadillo>
#include <limits>
#include <sstream>

namespace fs = boost::filesystem;

// Function to normalize an image to [0, 1]
cv::Mat normalizeImage(const cv::Mat& image) {
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F, 1.0 / 255.0);
    return normalized;
}

cv::Mat get_image_from_directory(const std::string& directory, const std::string& image_name) {
    // Get the directory path of the current script
    fs::path scriptDirectory = fs::current_path();

    // Combine the script directory with the provided directory name
    fs::path directoryPath = scriptDirectory / directory;

    // Check if the specified directory exists
    if (fs::is_directory(directoryPath)) {
        // Combine the directory path with the image name
        fs::path imagePath = directoryPath / image_name;

        // Check if the image file exists
        if (fs::is_regular_file(imagePath)) {
            // Load and return the image using OpenCV
            cv::Mat image = cv::imread(imagePath.string());
            return image;
        } else {
            std::cerr << "Image file '" << image_name << "' does not exist in directory '" << directory << "'." << std::endl;
        }
    } else {
        std::cerr << "Directory '" << directory << "' does not exist in the current path." << std::endl;
    }

    return cv::Mat();  // Return an empty image if the image does not exist
}

std::tuple<arma::mat, arma::mat, arma::mat> pinvFast(const arma::mat& A) {
    int m = A.n_rows;
    int n = A.n_cols;
    arma::mat U, S, V;

    if (m >= n) {
        arma::mat M1 = A.t() * A;
        arma::vec y1;
        arma::mat V1;
        arma::eig_sym(y1, V1, M1);

        double const_value = n * arma::max(y1) * std::numeric_limits<double>::epsilon();
        arma::uvec y2 = (y1 > const_value);
        int rA = arma::sum(y2);

        if (rA > 0) {
            arma::vec y3 = arma::abs(y1) % arma::conv_to<arma::vec>::from(y2);
            arma::vec s1 = arma::sqrt(y3);
            arma::uvec order = arma::sort_index(s1, "descend");
            V = V1.cols(order.head(rA));
            S = arma::diagmat(s1(order.head(rA)));
            U = A * V;
            U.each_row() /= s1(order.head(rA)).t();
        } else {
            // Handle the case where rA is 0
            U = arma::mat(m, 0, arma::fill::zeros); // Create an empty matrix filled with zeros
            S = arma::mat(0, 0, arma::fill::zeros); // Create an empty matrix filled with zeros
            V = arma::mat(n, 0, arma::fill::zeros); // Create an empty matrix filled with zeros
        }
    } else {
        arma::mat M1 = A * A.t();
        arma::vec y1;
        arma::mat U1;
        arma::eig_sym(y1, U1, M1);

        double const_value = m * arma::max(y1) * std::numeric_limits<double>::epsilon();
        arma::uvec y2 = (y1 > const_value);
        int rA = arma::sum(y2);

        if (rA > 0) {
            arma::vec y3 = arma::abs(y1) % arma::conv_to<arma::vec>::from(y2);
            arma::vec s1 = arma::sqrt(y3);
            arma::uvec order = arma::sort_index(s1, "descend");
            U = U1.cols(order.head(rA));
            S = arma::diagmat(s1(order.head(rA)));
            V = A.t() * U;
            V.each_row() /= s1(order.head(rA)).t();
        } else {
            // Handle the case where rA is 0
            U = arma::mat(m, 0, arma::fill::zeros); // Create an empty matrix filled with zeros
            S = arma::mat(0, 0, arma::fill::zeros); // Create an empty matrix filled with zeros
            V = arma::mat(n, 0, arma::fill::zeros); // Create an empty matrix filled with zeros
        }
    }
    return std::make_tuple(U, S, V);
}

void writeImageToTxt(const cv::Mat& image, const std::string& filename) {
    // Write the image data to a text file
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int row = 0; row < image.rows; ++row) {
            for (int col = 0; col < image.cols; ++col) {
                file << static_cast<double>(image.at<float>(row, col)) << ' ';
            }
            file << '\n';
        }
        file.close();
    } else {
        std::cerr << "Failed to open the output file for writing." << std::endl;
    }
}

int main() {
    const int numImg = 1110;
    const int numSCount = 6;
    const int imageWidth = 128;
    const int imageHeight = 128;

    // Load training images
    arma::mat X(imageWidth * imageHeight, numImg);
    arma::mat Y(imageWidth * imageHeight, numImg);

    for (int k = 1; k <= numImg; ++k) {
        std::string image_path = "image_free_noise";
        std::string image_name = "coast (" + std::to_string(k) + ").jpg";
        cv::Mat X1 = get_image_from_directory(image_path, image_name);
        cv::Mat X2 = normalizeImage(X1);
        cv::Mat X3;
        X2.convertTo(X3, CV_64F);
        X3 = X3.reshape(1, imageWidth * imageHeight);
        for (int i = 0; i < imageWidth * imageHeight; ++i) {
            X(i, k - 1) = X3.at<double>(0, i);
        }

        image_path = "image_with_noise";
        image_name = "coast (" + std::to_string(k) + ").jpg";
        cv::Mat Y1 = get_image_from_directory(image_path, image_name);
        cv::Mat Y2 = normalizeImage(Y1);
        cv::Mat Y3;
        Y2.convertTo(Y3, CV_64F);
        Y3 = Y3.reshape(1, imageWidth * imageHeight);
        for (int i = 0; i < imageWidth * imageHeight; ++i) {
            Y(i, k - 1) = Y3.at<double>(0, i);
        }
    }

    arma::vec numS(numSCount);
    numS.imbue([=]() { return std::rand() % numImg + 1; });

    // The code to display random images goes here

    // Check if matrix T is rank-deficient
    arma::mat T = Y.t() * Y;
    int rA = arma::rank(T);
    if (rA >= std::min(T.n_rows, T.n_cols)) {
        std::cout << "Matrix T is full rank." << std::endl;
    } else {
        std::cout << "Matrix T is rank-deficient. Rank(T) = " << rA << std::endl;
    }

    // Compute Filter F1 (using proposed method) and F2 (using pinv)
    arma::mat Yp1;
    auto start_time1 = std::clock();
    std::tie(std::ignore, std::ignore, Yp1) = pinvFast(Y);
    arma::mat F1 = X * Yp1;
    auto end_time1 = std::clock();

    arma::mat Yp2 = arma::pinv(Y);
    arma::mat F2 = X * Yp2;

    // Print the sizes of the variables
    std::cout << "Size of F1: " << F1.n_rows << " x " << F1.n_cols << std::endl;
    std::cout << "Size of Yp2: " << Yp2.n_rows << " x " << Yp2.n_cols << std::endl;
    std::cout << "Size of F2: " << F2.n_rows << " x " << F2.n_cols << std::endl;
    std::cout << "Size of Yp1: " << Yp1.n_rows << " x " << Yp1.n_cols << std::endl;

    // Perform analysis of proposed method vs command pinv
    double execution_time1 = static_cast<double>(end_time1 - start_time1) / CLOCKS_PER_SEC;
    std::cout << "Execution time to compute Moore-Penrose of Y using the proposed method = " << execution_time1 << " seconds" << std::endl;

    // Error calculation for F1 (proposed method) and F2 (pinv)
    double error_estimation_pm = arma::norm(Yp1 - Yp2, "fro");
    std::cout << "Error estimation of proposed method vs. pinv: " << error_estimation_pm << std::endl;

    // The code to display test images and perform denoising goes here

    return 0;
}
