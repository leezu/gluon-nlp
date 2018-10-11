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

// * Includes and definitions
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "CLI/CLI.hpp"      // command line parser
#include "cnpy.h"           // numpy
#include "range/v3/all.hpp" // Ranges TS https://github.com/ericniebler/range-v3/

using count_type = float;

// * Arguments
struct Arguments {
  float context_distribution_smoothing = 0.75;
  std::string cooccurrences = "cooccurrences.npz";
  std::string output = "ppmi.npz";
};

auto parseArgs(int argc, char **argv) {
  // Performance optimizations for writing to stdout
  std::ios::sync_with_stdio(false);

  Arguments args;
  CLI::App app(
      "Simple tool to compute PPMI matrix based on co-occurrence counts.");
  app.add_option("cooccurrence", args.cooccurrences,
                 "COO matrix of co-occurence counts.")
      ->check(CLI::ExistingPath)
      ->required();
  app.add_option("-o,--output", args.output,
                 "Output file name. Numpy .npz archive containing input for "
                 "alias method is written.");
  app.add_option(
      "-s,--context-distribution-smoothing",
      args.context_distribution_smoothing,
      "Context distribution smoothing. See Levy et al. 2015. Only active if > "
      "0. If active, resulting matrix won't be symmetric.");
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    std::exit(app.exit(e));
  }

  return args;
}

// * Input
std::tuple<std::vector<uint32_t>, std::vector<uint32_t>,
           std::vector<count_type>, bool>
readNumpyFiles(const Arguments &args) {
  cnpy::npz_t cooccurrences = cnpy::npz_load(args.cooccurrences);
  assert(cooccurrences["row"].word_size == sizeof(uint32_t));
  assert(cooccurrences["col"].word_size == sizeof(uint32_t));
  assert(cooccurrences["data"].word_size == sizeof(count_type));
  assert(cooccurrences["row"].shape.size() == 1);
  assert(cooccurrences["col"].shape.size() == 1);
  assert(cooccurrences["data"].shape.size() == 1);

  // "symmetric" flag is optional. Assume false if not present.
  bool symmetric = false;
  if (cooccurrences.find("symmetric") != cooccurrences.end()) {
    assert(cooccurrences["symmetric"].word_size == sizeof(bool));
    symmetric = *cooccurrences["symmetric"].data<bool>();
  }

  return {
      {cooccurrences["row"].data<uint32_t>(),
       cooccurrences["row"].data<uint32_t>() + cooccurrences["row"].shape[0]},
      {cooccurrences["col"].data<uint32_t>(),
       cooccurrences["col"].data<uint32_t>() + cooccurrences["col"].shape[0]},
      {cooccurrences["data"].data<count_type>(),
       cooccurrences["data"].data<count_type>() +
           cooccurrences["data"].shape[0]},
      symmetric};
}

// * PPMI
std::vector<count_type> computeMarginal(const std::vector<uint32_t> &row,
                                        const std::vector<uint32_t> &col,
                                        const std::vector<count_type> &data,
                                        bool symmetric) {
  std::vector<count_type> marginal(data.size());
  if (symmetric) {
    for (size_t i = 0; i < data.size(); i++) {
      marginal[row[i]] += data[i];
      marginal[col[i]] += data[i];
    }
  } else {
    for (size_t i = 0; i < data.size(); i++) {
      marginal[row[i]] += data[i];
    }
  }
  return marginal;
}

void ppmi(const Arguments &args, const std::vector<uint32_t> &row,
          const std::vector<uint32_t> &col, std::vector<count_type> &data,
          const std::vector<count_type> &marginal, bool symmetric) {
  if (args.context_distribution_smoothing > 0) {
    size_t size = data.size();
    for (size_t i = 0; i < size; i++) {
      auto log_data = std::log(data[i]);
      data[i] = log_data - std::log(marginal[row[i]]) -
                std::log(std::pow(marginal[col[i]],
                                  args.context_distribution_smoothing));
      if (symmetric) {
        // Must store previously implict elements now explicitly
        data.push_back(log_data -
                       std::log(std::pow(marginal[row[i]],
                                         args.context_distribution_smoothing)) -
                       std::log(marginal[col[i]]));
      }
    }
  } else {
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = std::log(data[i]) - std::log(marginal[row[i]]) -
                std::log(marginal[col[i]]);
    }
  }
}

// * Output
void writeNumpy(const Arguments &args, const std::vector<uint32_t> &row,
                const std::vector<uint32_t> &col,
                const std::vector<count_type> &data, bool symmetric) {

  assert((row.size() == data.size()) | (2 * row.size() == data.size()));
  assert((col.size() == data.size()) | (2 * col.size() == data.size()));
  cnpy::npz_save(args.output, "row", &row[0], {row.size()}, "w");
  cnpy::npz_save(args.output, "col", &col[0], {col.size()}, "a");
  cnpy::npz_save(args.output, "data", &data[0], {data.size()}, "a");
  cnpy::npz_save(args.output, "symmetric", &symmetric, {1}, "a");
}

// * Main
int main(int argc, char **argv) {
  auto args = parseArgs(argc, argv);
  auto [row, col, data, symmetric] = readNumpyFiles(args);
  auto marginal = computeMarginal(row, col, data, symmetric);
  ppmi(args, row, col, data, marginal, symmetric);
  writeNumpy(args, row, col, data, symmetric);
  return 0;
}
