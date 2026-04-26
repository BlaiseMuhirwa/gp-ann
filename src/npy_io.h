#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "defs.h"
#include <parlay/parallel.h>

namespace npy {

struct Header {
  std::string descr;
  std::vector<size_t> shape;
  size_t data_offset = 0;
  size_t element_size = 0;
};

inline Header ParseHeader(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open())
    throw std::runtime_error("Cannot open npy file: " + path);

  // Verify magic: \x93NUMPY
  char magic[6];
  in.read(magic, 6);
  if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' ||
      magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y')
    throw std::runtime_error("Invalid .npy magic: " + path);

  uint8_t major, minor;
  in.read(reinterpret_cast<char *>(&major), 1);
  in.read(reinterpret_cast<char *>(&minor), 1);

  uint32_t header_len;
  if (major == 1) {
    uint16_t hl;
    in.read(reinterpret_cast<char *>(&hl), 2);
    header_len = hl;
  } else if (major == 2) {
    in.read(reinterpret_cast<char *>(&header_len), 4);
  } else {
    throw std::runtime_error(
        "Unsupported npy version: " + std::to_string(major) + "." +
        std::to_string(minor));
  }

  std::string hdr(header_len, '\0');
  in.read(&hdr[0], header_len);

  Header result;
  result.data_offset = static_cast<size_t>(in.tellg());

  // Parse descr: find the value between quotes after 'descr':
  // Header looks like: {'descr': '|u1', ...}
  // We need to skip past the key quotes and colon to the value quotes.
  auto pos = hdr.find("descr");
  if (pos == std::string::npos)
    throw std::runtime_error("No descr in npy header");
  auto colon = hdr.find(':', pos);
  // Find opening quote of the value (after the colon)
  auto q1 = hdr.find('\'', colon);
  auto q1d = hdr.find('"', colon);
  char quote_char = '\'';
  if (q1d != std::string::npos && (q1 == std::string::npos || q1d < q1)) {
    q1 = q1d;
    quote_char = '"';
  }
  auto q2 = hdr.find(quote_char, q1 + 1);
  result.descr = hdr.substr(q1 + 1, q2 - q1 - 1);

  // Determine element size from dtype descriptor
  if (result.descr == "<u1" || result.descr == "|u1")
    result.element_size = 1;
  else if (result.descr == "<u2" || result.descr == "|u2")
    result.element_size = 2;
  else if (result.descr == "<u4" || result.descr == "|u4" ||
           result.descr == "<i4" || result.descr == "|i4" ||
           result.descr == "<f4" || result.descr == "|f4")
    result.element_size = 4;
  else if (result.descr == "<f8" || result.descr == "|f8")
    result.element_size = 8;
  else
    throw std::runtime_error("Unsupported npy dtype: " + result.descr);

  // Parse shape: find the tuple after 'shape':
  auto sp = hdr.find("shape");
  auto sc = hdr.find(':', sp);
  auto po = hdr.find('(', sc);
  auto pc = hdr.find(')', po);
  std::string shape_str = hdr.substr(po + 1, pc - po - 1);

  size_t i = 0;
  while (i < shape_str.size()) {
    while (i < shape_str.size() && (shape_str[i] == ' ' || shape_str[i] == ','))
      i++;
    if (i >= shape_str.size())
      break;
    size_t end = i;
    while (end < shape_str.size() && shape_str[end] >= '0' &&
           shape_str[end] <= '9')
      end++;
    if (end > i)
      result.shape.push_back(std::stoull(shape_str.substr(i, end - i)));
    i = end;
  }

  return result;
}

// Read a 2D uint8 npy file as a float32 PointSet (parallel, chunked).
// If max_rows > 0, only the first max_rows rows are read.
inline PointSet ReadUint8AsPointSet(const std::string &path,
                                    int64_t max_rows = -1) {
  Header hdr = ParseHeader(path);

  if (hdr.element_size != 1)
    throw std::runtime_error("Expected uint8 npy, got dtype=" + hdr.descr);
  if (hdr.shape.size() != 2)
    throw std::runtime_error("Expected 2D array in " + path);

  size_t n = hdr.shape[0];
  size_t d = hdr.shape[1];
  if (max_rows > 0 && static_cast<size_t>(max_rows) < n)
    n = max_rows;

  std::cout << "npy: reading " << n << " x " << d << " uint8 from " << path
            << std::endl;

  PointSet points;
  points.n = n;
  points.d = d;
  points.coordinates.resize(n * d);

  Timer timer;
  timer.Start();

  size_t num_chunks = parlay::num_workers();
  size_t rows_per_chunk = (n + num_chunks - 1) / num_chunks;

  parlay::parallel_for(
      0, num_chunks,
      [&](size_t chunk) {
        size_t row_begin = chunk * rows_per_chunk;
        size_t row_end = std::min(n, (chunk + 1) * rows_per_chunk);
        if (row_begin >= row_end)
          return;

        size_t num_bytes = (row_end - row_begin) * d;
        std::vector<uint8_t> buffer(num_bytes);

        std::ifstream in(path, std::ios::binary);
        in.seekg(hdr.data_offset + row_begin * d);
        in.read(reinterpret_cast<char *>(buffer.data()), num_bytes);

        size_t out_offset = row_begin * d;
        for (size_t i = 0; i < num_bytes; ++i) {
          points.coordinates[out_offset + i] = static_cast<float>(buffer[i]);
        }
      },
      1);

  std::cout << "npy: read took " << timer.Stop() << "s" << std::endl;
  return points;
}

// Ground truth IDs from a 2D uint32 npy file.
struct GroundTruthIds {
  std::vector<uint32_t> ids; // flat: num_queries * num_neighbors
  size_t num_queries;
  size_t num_neighbors;

  uint32_t Get(size_t q, size_t j) const { return ids[q * num_neighbors + j]; }
};

inline GroundTruthIds ReadUint32_2D(const std::string &path) {
  Header hdr = ParseHeader(path);

  if (hdr.element_size != 4)
    throw std::runtime_error("Expected uint32 npy, got dtype=" + hdr.descr);
  if (hdr.shape.size() != 2)
    throw std::runtime_error("Expected 2D array in " + path);

  GroundTruthIds result;
  result.num_queries = hdr.shape[0];
  result.num_neighbors = hdr.shape[1];
  result.ids.resize(result.num_queries * result.num_neighbors);

  std::cout << "npy: reading ground truth " << result.num_queries << " x "
            << result.num_neighbors << " from " << path << std::endl;

  std::ifstream in(path, std::ios::binary);
  in.seekg(hdr.data_offset);
  in.read(reinterpret_cast<char *>(result.ids.data()),
          result.ids.size() * sizeof(uint32_t));

  return result;
}

} // namespace npy
