#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <poppler-document.h>
#include <poppler-page.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>
#include <mutex>
#include <vector>

using json = nlohmann::json;
std::mutex jsonMutex; // Mutex for thread-safe JSON access

// Preprocess image using OpenCV
std::string preprocessAndOCR(const std::string &imagePath) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << imagePath << std::endl;
        return "";
    }

    // Apply preprocessing: thresholding, noise removal
    cv::Mat processedImg;
    cv::threshold(img, processedImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::imwrite("temp_processed.png", processedImg); // Save processed image

    // Perform OCR on the processed image
    tesseract::TessBaseAPI ocr;
    if (ocr.Init(nullptr, "eng")) {
        std::cerr << "Could not initialize Tesseract." << std::endl;
        return "";
    }

    Pix *image = pixRead("temp_processed.png");
    if (!image) {
        std::cerr << "Failed to open processed image file." << std::endl;
        return "";
    }

    ocr.SetImage(image);
    std::string text = ocr.GetUTF8Text();
    ocr.End();
    pixDestroy(&image);

    // Clean up temporary file
    std::remove("temp_processed.png");
    return text;
}

// Extract text from PDF
std::string extractPDFText(const std::string &pdfPath) {
    auto doc = poppler::document::load_from_file(pdfPath);
    if (!doc) {
        std::cerr << "Failed to open PDF file: " << pdfPath << std::endl;
        return "";
    }

    std::string pdfText;
    for (int i = 0; i < doc->pages(); ++i) {
        auto page = doc->create_page(i);
        if (page) {
            pdfText += page->text().to_latin1();
        }
    }
    return pdfText;
}

// Process individual file
void processFile(const std::string &file, json &result) {
    std::string extension = file.substr(file.find_last_of('.') + 1);
    std::string extractedText;

    if (extension == "pdf") {
        extractedText = extractPDFText(file);
    } else {
        extractedText = preprocessAndOCR(file);
    }

    if (!extractedText.empty()) {
        // Lock mutex to safely modify JSON
        std::lock_guard<std::mutex> lock(jsonMutex);
        result[file] = {
            {"type", extension == "pdf" ? "PDF" : "Image"},
            {"text", extractedText}
        };
    }
}

// Multithreaded document processing
void processDocuments(const std::vector<std::string> &files, const std::string &outputJson) {
    json result;
    std::vector<std::thread> threads;

    for (const auto &file : files) {
        threads.emplace_back(processFile, file, std::ref(result));
    }

    // Wait for all threads to complete
    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Save result to JSON file
    std::ofstream outFile(outputJson);
    if (!outFile) {
        std::cerr << "Failed to write JSON file: " << outputJson << std::endl;
        return;
    }
    outFile << result.dump(4); // Pretty print with indentation
    std::cout << "Data saved to " << outputJson << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <output_json> <files...>" << std::endl;
        return 1;
    }

    std::string outputJson = argv[1];
    std::vector<std::string> files(argv + 2, argv + argc);

    processDocuments(files, outputJson);

    return 0;
}
