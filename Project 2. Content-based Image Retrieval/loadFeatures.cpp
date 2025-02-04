#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring> // For strcmp

using namespace std;

// Function to split a string by a delimiter
vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

char *removePrefix(const char *targetFilename) {
    const char prefix[] = "olympus/";
    size_t prefix_len = strlen(prefix);

    // Check if the targetFilename starts with "olympus/"
    if (strncmp(targetFilename, prefix, prefix_len) == 0) {
        return strdup(targetFilename + prefix_len); // Allocate new memory for the substring
    } else {
        return strdup(targetFilename); // Allocate new memory for the original string
    }
}

// Function to extract the feature vector for a target image
vector<float> extractFeatureVector(const char *targetFilename, const char *featuresFile) {
    vector<float> features;
    ifstream file(featuresFile);
    string line;

    if (!file.is_open()) {
        cerr << "Error: Could not open the features file!" << endl;
        return features;
    }

    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        string filename = tokens[0];

        // Check if the current row corresponds to the target image
        if (filename == targetFilename) {
            // Extract the feature values
            for (size_t i = 1; i < tokens.size(); ++i) {
                features.push_back(stof(tokens[i]));
            }
            break; // Stop searching once the target image is found
        }
    }

    file.close();

    if (features.empty()) {
        cerr << "Error: Target image not found in the features file!" << endl;
    }

    return features;
}


int main() {
    char targetFilename[256] = "olympus/pic.0893.jpg"; // Target image filename
    char featuresFile[256] = "ResNet18_olym.csv"; // Path to the CSV file

    // Extract the feature vector for the target image
    vector<float> features = extractFeatureVector(removePrefix(targetFilename), featuresFile);

    // Print the extracted feature vector
    if (!features.empty()) {
        cout << "Feature vector for " << targetFilename << ":" << endl;
        for (float value : features) {
            cout << value << " ";
        }
        cout << endl;
    }

    return 0;
}