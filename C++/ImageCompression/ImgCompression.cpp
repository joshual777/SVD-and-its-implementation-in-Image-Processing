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

cv::Mat openImageInSameDirectory(const std::string& filename) {
    fs::path scriptDirectory = fs::current_path();
    fs::path imagePath = scriptDirectory;
    imagePath.append(filename);
    return cv::imread(imagePath.string());
}

std::pair<int, int> getImageDimensions(const cv::Mat& image) {
    return std::make_pair(image.cols, image.rows);
}

/*
cv::Mat normalizeImage(const cv::Mat& image) {
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F, 1.0 / 255.0);
    return normalized;
}
*/

cv::Mat normalizeImage(const cv::Mat& image) {
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F, 1.0 / 255.0);

    // Truncate values to 6 decimal places
    for (int i = 0; i < normalized.rows; ++i) {
        for (int j = 0; j < normalized.cols; ++j) {
            float value = normalized.at<float>(i, j);
            normalized.at<float>(i, j) = static_cast<float>(static_cast<int>(value * 1e6)) / 1e6;
        }
    }

    return normalized;
}


void storeImageInSameDirectory(const cv::Mat& image, const std::string& filename) {
    fs::path scriptDirectory = fs::current_path();
    fs::path imagePath = scriptDirectory;
    imagePath.append(filename);
    cv::imwrite(imagePath.string(), image);
}

arma::mat cvMatToArmaMat(const cv::Mat& cvMatrix) {
    int rows = cvMatrix.rows;
    int cols = cvMatrix.cols;

    arma::mat armaMatrix(rows, cols, arma::fill::zeros);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            armaMatrix(i, j) = static_cast<double>(cvMatrix.at<float>(i, j)); // Convert to double if necessary
        }
    }

    return armaMatrix;
}

cv::Mat armaMatToCvMat(const arma::mat& armaMatrix) {
    int rows = armaMatrix.n_rows;
    int cols = armaMatrix.n_cols;

    cv::Mat cvMatrix(rows, cols, CV_32F);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cvMatrix.at<float>(i, j) = static_cast<float>(armaMatrix(i, j)); // Convert to float if necessary
        }
    }

    return cvMatrix;
}

void writeMatToCSV(const arma::mat& mat, const std::string& filename) {
    mat.save(filename, arma::csv_ascii);
}

arma::mat readMatFromCSV(const std::string& filename) {
    arma::mat mat;
    mat.load(filename, arma::csv_ascii);
    return mat;
}

cv::Size getFileMatrixDimensions(const std::string& filename) {
    // Open the file for reading
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return cv::Size(0, 0); // Return an empty size to indicate an error
    }

    // Read CSV data into a vector of vectors (rows x columns)
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        double value;
        while (ss >> value) {
            row.push_back(value);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
        data.push_back(row);
    }

    // Calculate the dimensions (rows x columns)
    int rows = static_cast<int>(data.size());
    int cols = (rows > 0) ? static_cast<int>(data[0].size()) : 0;

    if (rows > 0 && cols > 0) {
        return cv::Size(cols, rows);
    } else {
        std::cerr << "Error: Could not determine matrix dimensions from file " << filename << std::endl;
        return cv::Size(0, 0); // Return an empty size to indicate an error
    }
}

std::tuple<arma::mat, arma::mat, arma::mat> svdCPP(const arma::mat& A) {
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
    // Calculate 'Ap' using Armadillo functions
    arma::mat Ap = V * arma::inv(S) * u.T();

    return Ap;
}

void printFirst20Pixels(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Image is empty." << std::endl;
        return;
    }

    // Ensure the image is not too small (at least 20 pixels)
    if (image.rows * image.cols < 20) {
        std::cerr << "Image is too small to print 20 pixels." << std::endl;
        return;
    }

    // Print the first 20 pixel values (0-255 for each channel)
    for (int i = 0; i < 20; ++i) {
        int row = i / image.cols;
        int col = i % image.cols;
        cv::Vec3b pixel = image.at<cv::Vec3b>(row, col);

        std::cout << "Pixel " << i << ": B = " << static_cast<int>(pixel[0])
                  << ", G = " << static_cast<int>(pixel[1])
                  << ", R = " << static_cast<int>(pixel[2]) << std::endl;
    }
}


int main() {
    
    cv::Mat I1 = openImageInSameDirectory("img1.jpg");

    printFirst20Pixels(I1);

    cv::Mat I = normalizeImage(I1);

    storeImageInSameDirectory(I, "normalized_image.jpg");

    writeMatToCSV(cvMatToArmaMat(I), "originalImage.txt");

    cv::Size fileDimensions = getFileMatrixDimensions("originalImage.txt");
    std::cout << "Dimensions of originalImage.txt: " << fileDimensions.width << " x " << fileDimensions.height << std::endl;

    int r = 10;
    std::tuple<arma::mat, arma::mat, arma::mat> result = svdCPP(cvMatToArmaMat(I));
    arma::mat U, S, V;
    U = std::get<0>(result);
    S = std::get<1>(result);
    V = std::get<2>(result);
    arma::mat Ur = U.cols(0, r - 1);
    arma::mat Sr = S.submat(0, 0, r - 1, r - 1);
    arma::mat Vr = V.cols(0, r - 1);
    arma::mat D = Ur * Sr;
    arma::mat C = Vr.t();


    writeMatToCSV(D, "D.txt");
    writeMatToCSV(C, "C.txt");

    std::cout << "D Rows = " << D.n_cols << ", D Columns = " << D.n_rows << std::endl;
    std::cout << "C Rows = " << C.n_cols << ", C Columns = " << C.n_rows << std::endl;

    cv::Size DDimensions = getFileMatrixDimensions("D.txt");
    std::cout << "Dimensions of D.txt: " << DDimensions.width << " x " << DDimensions.height << std::endl;

    cv::Size CDimensions = getFileMatrixDimensions("C.txt");
    std::cout << "Dimensions of C.txt: " << CDimensions.width << " x " << CDimensions.height << std::endl;

    arma::mat D1 = readMatFromCSV("D.txt");
    arma::mat C1 = readMatFromCSV("C.txt");

    // Reconstruct the compressed image
    arma::mat Ic1 = D1 * C1;

    // Display the reconstructed image
    cv::Mat Ic = armaMatToCvMat(Ic1);
    storeImageInSameDirectory(Ic, "compressed_image.jpg");

    // Get the dimensions of the image
    int width = Ic.cols;
    int height = Ic.rows;

    // Print the dimensions
    std::cout << "Width: " << width << " pixels" << std::endl;
    std::cout << "Height: " << height << " pixels" << std::endl;
    cv::imshow("Compressed Image", Ic);

    cv::waitKey(0);

    return 0;
}

