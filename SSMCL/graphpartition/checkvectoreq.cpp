#include <iostream>
#include <fstream>
#include <string>
#include <vector>


std::vector<int64_t> readFileToVector(const std::string& fileName) {
    std::vector<int64_t> result;
    std::ifstream file(fileName);

    if (!file) {
        std::cerr << "Error: Unable to open file: " << fileName << std::endl;
        return result;
    }

    std::string line;
    while (std::getline(file, line)) {
        try {
            result.push_back(std::stol(line));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Invalid integer value on line: " << line << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Integer value out of range on line: " << line << std::endl;
        }
    }

    file.close();
    return result;
}

int main(int argc, char ** argv) {
    std::string file1(argv[1]);
    std::string file2(argv[2]);
    std::cout << "compare " << file1 << ", " << file2 << std::endl;
    std::vector<int64_t> results = readFileToVector(file1);
    std::vector<int64_t> tmpgraph = readFileToVector(file2);
    if(results.size() != tmpgraph.size()) {
        std::cout << "dim not match, results " << results.size() << " tmpgraph, " << tmpgraph.size() << std::endl; 
        exit(0);
    }
    int64_t diff(0);
    for(int64_t i=0; i<results.size(); i++){
        if(results[i] != tmpgraph[i]) {
            printf("results %ld tmpgraph %ld \n", results[i], tmpgraph[i]);
            diff++;
        }
    }
    std::cout << "diff:" << diff << ", tot:" << results.size() << std::endl;

    return 0;
}