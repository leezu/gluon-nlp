// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sentencepiece_processor.h>

namespace py = pybind11;

class SentencePieceWrap {
public:
  SentencePieceWrap(const std::string &filename) {
    auto status = model.Load(filename);
    if (!status.ok()) {
      throw std::runtime_error("Failed loading SentencePieceProcessor");
    }
  };

  int size() { return model.GetPieceSize(); }

  void set_tokens(std::vector<std::string> idx_to_token) {
    this->idx_to_token = idx_to_token;
  }

  std::vector<std::vector<int>> encode(std::vector<int> idxs) {
    std::vector<std::vector<int>> sampled;
    for (int idx : idxs) {
      std::vector<int> idx_sampled;
      model.Encode(idx_to_token.at(idx), &idx_sampled);
      sampled.push_back(idx_sampled);
    }
    return sampled;
  }

  std::vector<std::vector<int>> sample_encode(const std::vector<int> &idxs,
                                              const int nbest_size,
                                              const float alpha) {
    std::vector<std::vector<int>> sampled;
    for (int idx : idxs) {
      std::vector<int> idx_sample;
      model.SampleEncode(idx_to_token.at(idx), nbest_size, alpha, &idx_sample);
      sampled.push_back(idx_sample);
    }
    return sampled;
  }

  auto sample_skipgram(const std::vector<int> &idxs, const int nbest_size,
                       const float alpha, const int offset) {
    std::vector<float> data;
    std::vector<int> row, col;

    {
      py::gil_scoped_release release;

      std::vector<std::vector<int>> sampled =
          sample_encode(idxs, nbest_size, alpha);
      for (unsigned int i = 0; i < idxs.size(); i++) {
        int idx = idxs[i];
        std::vector<int> &idx_sample = sampled[i];
        auto size = idx_sample.size();

        data.push_back(1.0 / (size + 1));
        row.push_back(i);
        col.push_back(idx);

        for (int s : idx_sample) {
          data.push_back(1.0 / (size + 1));
          row.push_back(i);
          col.push_back(s + offset);
        }
      }
    }

    auto data_np = py::array(data.size(), data.data());
    auto row_np = py::array(row.size(), row.data());
    auto col_np = py::array(col.size(), col.data());
    return std::make_tuple(data_np, row_np, col_np);
  }

private:
  sentencepiece::SentencePieceProcessor model;
  std::vector<std::string> idx_to_token;
};

PYBIND11_MODULE(sentencepiecewrap, m) {
  m.doc() = "SentencePiece nogil wrapper";
  py::class_<SentencePieceWrap>(m, "SentencePieceWrap")
      .def(py::init<const std::string &>())
      .def("__len__", &SentencePieceWrap::size)
      .def("set_tokens", &SentencePieceWrap::set_tokens)
      .def("encode", &SentencePieceWrap::encode,
           py::call_guard<py::gil_scoped_release>())
      .def("sample_skipgram", &SentencePieceWrap::sample_skipgram)
      .def("sample_encode", &SentencePieceWrap::sample_encode,
           py::call_guard<py::gil_scoped_release>());
}
