#ifndef _INCLUDE_SAMPLING_
#define _INCLUDE_SAMPLING_

void c_construct_random_propagation_matrix(
    const float * const in_data, const int * const in_indices,
    const int * const in_indptr, const long unsigned int in_indptr_size,
    const int n_samples, std::vector<float> &out_data,
    std::vector<int> &out_indices, std::vector<int> &out_indptr) noexcept;

#endif