#ifndef DUMMY_API_H_
#define DUMMY_API_H_

#include <iostream>
#include <vector>

namespace dummyapi {
// Data type of model's input and output.
struct DataInfo {
  enum Type {
    Float32 = 0,
    Uint8 = 1,
    Int8 = 2,
    Float16 = 3,
    Int32 = 4,
    Int64 = 5,
  };

  DataInfo(Type t, int s) {
    type = t;
    length = s;
  }

  Type type;
  int length;
};

// Initialize the dummy api with a given model.
void InitializeBackend(const std::string& model_file_path) {
  std::cout << "Dummy Backend initialized.\n";
}

// Sets inputs for a sample before inferencing.
void SetInputs(const std::vector<void*>& inputs) {
  std::cout << "Get a new input to the Dummy Backend.\n";
}

// Runs inference for a sample.
void Run() { std::cout << "Running a new inference with the Dummy Backend.\n"; }

// Returns the result after inferencing.
std::vector<std::vector<uint8_t>> GetOutputs() {
  return std::vector<std::vector<uint8_t>>(1, std::vector<uint8_t>(16, 2));
}

// Returns a dummy input format required by the model.
std::vector<DataInfo> GetInputFormat() {
  return {DataInfo(DataInfo::Float32, 1024)};
}

// Returns a dummy output format produced by the model.
std::vector<DataInfo> GetOutputFormat() {
  return {DataInfo(DataInfo::Float32, 4)};
}

}  // namespace dummyapi
#endif  // DUMMY_API_H_
