#ifndef _INCLUDE_SAMPLING_
#define _INCLUDE_SAMPLING_

#include <cstdint>


void c_construct_random_propagation_matrix(
    const float * const in_data, const int32_t * const in_indices,
    const int32_t * const in_indptr, const long unsigned int in_indptr_size,
    const int n_samples, std::vector<float> &out_data,
    std::vector<int32_t> &out_indices, std::vector<int32_t> &out_indptr) noexcept;

#endif