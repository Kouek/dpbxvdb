#include "downsample.cuh"

#include <thrust/device_vector.h>

#include <glm/gtc/matrix_transform.hpp>

std::vector<float> dpbxvdb::downsample(const thrust::device_vector<float> &d_src,
                                       DownsamplePolicy policy, const VDBInfo &vdbInfo) {
    thrust::device_vector<float> d_downsampled(
        vdbInfo.brickPerVol.x * vdbInfo.brickPerVol.y * vdbInfo.brickPerVol.z, 0.f);
    auto voxPerBrick = vdbInfo.dims[0];
    auto voxPerVol = vdbInfo.voxPerVol;
    auto brickPerVol = vdbInfo.brickPerVol;
    auto krnlFn = [src = thrust::raw_pointer_cast(d_src.data()),
                   dst = thrust::raw_pointer_cast(d_downsampled.data()), voxPerBrick, voxPerVol,
                   brickPerVol, policy] __device__(CoordTy idx3, IDTy idx) {
        auto vMin = voxPerBrick * idx3;
        auto vMax = voxPerBrick * (idx3 + glm::one<decltype(idx3)>());
        auto v = 0.f;
        for (auto z = vMin.z; z < vMax.z; ++z) {
            if (z >= voxPerVol.z)
                break;
            for (auto y = vMin.y; y < vMax.y; ++y) {
                if (y >= voxPerVol.y)
                    break;
                for (auto x = vMin.x; x < vMax.x; ++x) {
                    if (x >= voxPerVol.x)
                        break;
                    auto vIdx = (z * voxPerVol.y + y) * voxPerVol.x + x;
                    switch (policy) {
                    case DownsamplePolicy::Avg:
                        v += src[vIdx];
                        break;
                    case DownsamplePolicy::Max:
                        v = glm::max(src[vIdx], v);
                        break;
                    }
                }
            }
        }
        if (policy == DownsamplePolicy::Avg)
            v /= (float)voxPerBrick * voxPerBrick * voxPerBrick;
        dst[idx] = v;
    };
    dim3 threadPerBlock{8, 8, 8};
    dim3 blockPerGrid{
        (static_cast<decltype(dim3::x)>(brickPerVol.x) + threadPerBlock.x - 1) / threadPerBlock.x,
        (static_cast<decltype(dim3::x)>(brickPerVol.y) + threadPerBlock.y - 1) / threadPerBlock.y,
        (static_cast<decltype(dim3::x)>(brickPerVol.z) + threadPerBlock.z - 1) / threadPerBlock.z};
    parallelExec3D<<<blockPerGrid, threadPerBlock>>>(krnlFn, brickPerVol);

    std::vector<float> ret(d_downsampled.size());
    thrust::copy(d_downsampled.begin(), d_downsampled.end(), ret.begin());
    return ret;
}
