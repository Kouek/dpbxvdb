#ifndef KOUEK_DPBXVDB_KERNELS_DOWNSAMPLE_H
#define KOUEK_DPBXVDB_KERNELS_DOWNSAMPLE_H

#include "dpbxvdb.cuh"

#include <thrust/device_vector.h>

namespace dpbxvdb {

enum class DownsamplePolicy : uint8_t { Avg = 0, Max };

std::vector<float> downsample(const thrust::device_vector<float> &d_src, DownsamplePolicy policy,
                              const VDBInfo &vdbInfo);

} // namespace dpbxvdb

#endif // !KOUEK_DPBXVDB_KERNELS_DOWNSAMPLE_H
