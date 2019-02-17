#include <vector>
#include <random>
#include <iterator>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <cstdint>


std::unordered_set<int> sample_permutation(const long unsigned int L, const long unsigned int N) {
  // Create random array with size N from range [0, L)
  // It samples random number multiple times to sample without replacement.
  // Time complexity should be relative low since L >> N in most cases.
  if (L <= N) {
    std::unordered_set<int> s;
    for (long unsigned int i = 0; i < L; ++i) {
      s.insert(i);
    }
    return s;
  }
  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<const int> uni(0, L - 1);
  std::unordered_set<int> s;
  do {
    auto random_integer = uni(rng);
    const auto search = s.find(random_integer);
    if (search == s.cend()){
      s.insert(random_integer);
    }
  } while (s.size() < N);
  return s;
}


void c_construct_random_propagation_matrix(
    const float * const in_data, const int32_t * const in_indices,
    const int32_t * const in_indptr, const long unsigned int in_indptr_size,
    const float * const in_diags, const bool * const mask, const int n_samples,
    std::vector<float> &out_data, std::vector<int32_t> &out_indices,
    std::vector<int32_t> &out_indptr, std::vector<float> &out_full_data,
    std::vector<int32_t> &out_full_indices, std::vector<int32_t> &out_full_indptr) noexcept{
  const auto expected_size = n_samples * (in_indptr_size - 1);
  out_data.reserve(expected_size);
  out_indices.reserve(expected_size);
  out_indptr.reserve(in_indptr_size);

  out_full_data.reserve(expected_size);
  out_full_indices.reserve(expected_size);
  out_full_indptr.reserve(in_indptr_size);

  out_indptr.push_back(0);
  out_full_indptr.push_back(0);

  for (int i = 0; i < in_indptr_size - 1; ++i) {
    if (!mask[i]) {
      continue;
    }
    const int start = in_indptr[i];
    const int end = in_indptr[i + 1];
    const auto sample_indices = sample_permutation(end - start, n_samples);
    // #neighbors / #samples according to the original paper.
    const float n_D = (end - start) / float(sample_indices.size() + 1);
    bool flg = true;
    for (int s = 0; s < (end - start); ++s) {
      if (flg && in_indices[start + s] > i) {
        flg = false;
        out_indices.push_back(i);
        out_data.push_back(in_diags[i] * n_D);
        out_full_indices.push_back(i);
        out_full_data.push_back(in_diags[i]);
      }
      const auto search = sample_indices.find(s);
      if (search != sample_indices.cend()){
        out_indices.push_back(in_indices[start + s]);
        out_data.push_back(in_data[start + s] * n_D);
      }
      out_full_indices.push_back(in_indices[start + s]);
      out_full_data.push_back(in_data[start + s]);
    }
    if (flg) {
      out_indices.push_back(i);
      out_data.push_back(in_diags[i] * n_D);
      out_full_indices.push_back(i);
      out_full_data.push_back(in_diags[i]);
    }
    out_indptr.push_back(out_indptr.back() + sample_indices.size() + 1);
    out_full_indptr.push_back(out_full_indptr.back() + end - start + 1);
  }

  out_data.shrink_to_fit();
  out_indices.shrink_to_fit();

  out_full_data.shrink_to_fit();
  out_full_indices.shrink_to_fit();
}
