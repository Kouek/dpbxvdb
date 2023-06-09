#ifndef KOUEK_DPBXVDB_RESAMPLE_H
#define KOUEK_DPBXVDB_RESAMPLE_H

#include "dpbxvdb.cuh"

#include <vector>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

namespace dpbxvdb {

thrust::device_vector<float> loadByAxisTransform(const std::vector<float> &src,
                                                 const CoordTy &oldVoxPerVol,
                                                 const AxisTransform &axisTr);
void resample(const thrust::device_vector<float> &d_src, const VDBInfo &vdbInfo,
              const VDBDeviceData &vdbDat);
void resampleDepth(const VDBInfo &vdbInfo, const VDBDeviceData &vdbDat);

} // namespace dpbxvdb

#endif // !KOUEK_DPBXVDB_RESAMPLE_H
