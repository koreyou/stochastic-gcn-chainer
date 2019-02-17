#include <vector>
#include <random>
#include <iterator>
#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <cstdint>


std::vector<int> sample_permutation(const long unsigned int L, const long unsigned int N) {
  // Create random array with size N from range [0, L)
  // It samples random number multiple times to sample without replacement.
  // Time complexity should be relative low since L >> N in most cases.
  if (L <= N) {
    std::vector<int> v(L);
    std::iota(v.begin(), v.end(), 0);
    return v;
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
  std::vector<int> v;
  v.reserve(N);
  v.insert(v.end(), s.begin(), s.end());
  std::sort(v.begin(), v.end());
  return v;
}


void c_construct_random_propagation_matrix(
    const float * const in_data, const int32_t * const in_indices,
    const int32_t * const in_indptr, const long unsigned int in_indptr_size,
    const int n_samples, std::vector<float> &out_data,
    std::vector<int32_t> &out_indices, std::vector<int32_t> &out_indptr) noexcept{
  const auto expected_size = n_samples * (in_indptr_size - 1);
  out_data.reserve(expected_size);
  out_indices.reserve(expected_size);
  out_indptr.reserve(in_indptr_size);

  out_indptr.push_back(0);

  for (int i = 0; i < in_indptr_size - 1; ++i) {
    const int start = in_indptr[i];
    const int end = in_indptr[i + 1];
    const auto sample_indices = sample_permutation(end - start, n_samples);
    // #neighbors / #samples according to the original paper.
    const float n_D = (end - start) / float(sample_indices.size());
    for (auto s : sample_indices){
        out_indices.push_back(in_indices[start + s]);
        out_data.push_back(in_data[start + s] * n_D);
    }
    out_indptr.push_back(out_indptr.back() + sample_indices.size());
  }
  out_data.shrink_to_fit();
  out_indices.shrink_to_fit();
}
