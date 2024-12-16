#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <json/json.h>

// Function to remove whitespace and trim the string
std::string cleanWhitespace(const std::string &input) {
    std::string result = std::regex_replace(input, std::regex("^\\s+|\\s+$"), "");
    result = std::regex_replace(result, std::regex("\\s+"), " ");
    return result;
}

// Function to abbreviate words (can be expanded)
std::string abbreviate(const std::string &word) {
    // Add your custom abbreviation rules here
    if (word == "example") return "ex";
    if (word == "information") return "info";
    if (word == "approximate") return "approx";
    return word; // Return original word if no abbreviation rule matches
}

// Function to remove a letter from a 7+ letter word while keeping phonetic similarity
std::string shortenPhonetically(const std::string &word) {
    if (word.length() < 7) return word;
    // Example: Removing vowels except the first one
    std::string result;
    bool firstVowel = true;
    for (char c : word) {
        if (std::string("aeiouAEIOU").find(c) != std::string::npos) {
            if (firstVowel) {
                result += c;
                firstVowel = false;
            }
        } else {
            result += c;
        }
    }
    return result;
}

// Function to process the text
std::string processText(const std::string &input) {
    std::string result;
    std::istringstream stream(input);
    std::string word;

    while (stream >> word) {
        // Remove commas
        word = std::regex_replace(word, std::regex(","), "");

        // Abbreviate
        word = abbreviate(word);

        // Shorten phonetically if 7+ letters
        word = shortenPhonetically(word);

        // Append to result
        result += word + " ";
    }

    return cleanWhitespace(result);
}

// Save processed data to a JSON file
void saveToJson(const std::string &data, const std::string &filename) {
    Json::Value root;
    root["processed_text"] = data;

    std::ofstream file(filename, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    file << root;
    file.close();
    std::cout << "Data saved to JSON file: " << filename << std::endl;
}

int main() {
    // Input text (for testing)
    std::string inputText = "This,    is an example of   input text that needs to be cleaned, abbreviated, and processed.";

    // Process the text
    std::string cleanedText = processText(inputText);

    // Output the cleaned text
    std::cout << "Processed Text: " << cleanedText << std::endl;

    // Save to JSON
    saveToJson(cleanedText, "output.json");

    return 0;
}
