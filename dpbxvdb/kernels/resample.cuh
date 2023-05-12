#ifndef KOUEK_DPBXVDB_RESAMPLE_H
#define KOUEK_DPBXVDB_RESAMPLE_H

#include "dpbxvdb.cuh"

#include <vector>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

namespace dpbxvdb {

void resample(const thrust::device_vector<float> &d_src, float threshold, const VDBInfo &vdbInfo,
              const VDBDeviceData &vdbDat);

}

#endif // !KOUEK_DPBXVDB_RESAMPLE_H
