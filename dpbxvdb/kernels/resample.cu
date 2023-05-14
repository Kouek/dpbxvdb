#include "resample.cuh"

#include <glm/gtc/matrix_transform.hpp>

void dpbxvdb::resample(const thrust::device_vector<float> &d_src, const VDBInfo &vdbInfo,
                       const VDBDeviceData &vdbDat) {
    auto apronWidAndDep = static_cast<CoordValTy>(vdbInfo.apronWidAndDep);
    auto voxPerBrick = vdbInfo.dims[0];
    auto minDepIdx = vdbInfo.minDepIdx;
    auto maxDepIdx = vdbInfo.maxDepIdx;
    auto krnlFn = [src = thrust::raw_pointer_cast(d_src.data()), vdbInfo, vdbDat, apronWidAndDep,
                   voxPerBrick, minDepIdx, maxDepIdx] __device__(CoordTy aIdx3, IDTy aIdx) {
        CoordTy brickIdx3{aIdx3.x / vdbInfo.voxPerAtlasBrick, aIdx3.y / vdbInfo.voxPerAtlasBrick,
                          aIdx3.z / vdbInfo.voxPerAtlasBrick};
        auto aMin = vdbInfo.voxPerAtlasBrick * brickIdx3 + apronWidAndDep;
        auto vIdx3 = aIdx3 - aMin; // vox idx3 in Brick Space

        auto depSign = glm::zero<CoordTy>();
#pragma unroll
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (vIdx3[xyz] == minDepIdx)
                depSign[xyz] = 1;
            else if (vIdx3[xyz] == maxDepIdx)
                depSign[xyz] = -1;

        auto nodeID = vdbDat.atlasMaps[vdbInfo.AtlastBrickIdx32ID(brickIdx3)].node;
        if (nodeID == UndefID)
            return;

        auto vMin = vdbDat.nodePools[0][Node::ID2Idx(nodeID)].idx3;
        vIdx3 =
            glm::clamp(vMin + vIdx3, glm::zero<CoordTy>(),
                       vdbInfo.voxPerVol - static_cast<CoordValTy>(1)); // vox idx3 in Volume Space
        if (vdbInfo.useDPBX && (depSign.x | depSign.y | depSign.z) != 0) {
            auto pos = vIdx3 + apronWidAndDep * depSign;
            CoordValTy t = 0;
            while (true) {
                auto val = src[(pos.z * vdbInfo.voxPerVol.y + pos.y) * vdbInfo.voxPerVol.x + pos.x];
                if (val >= vdbInfo.thresh)
                    break;
                pos += depSign;
                ++t;
                if (t >= voxPerBrick)
                    break;
            }
            surf3Dwrite(static_cast<float>(t), vdbDat.atlasSurf, sizeof(float) * aIdx3.x, aIdx3.y,
                        aIdx3.z);
            return;
        }

        auto vIdx = (vIdx3.z * vdbInfo.voxPerVol.y + vIdx3.y) * vdbInfo.voxPerVol.x + vIdx3.x;
        auto v = src[vIdx];
        surf3Dwrite(v, vdbDat.atlasSurf, sizeof(float) * aIdx3.x, aIdx3.y, aIdx3.z);
    };
    dim3 threadPerBlock{8, 8, 8};
    dim3 blockPerGrid{
        (static_cast<decltype(dim3::x)>(vdbInfo.voxPerAtlas.x) + threadPerBlock.x - 1) /
            threadPerBlock.x,
        (static_cast<decltype(dim3::x)>(vdbInfo.voxPerAtlas.y) + threadPerBlock.y - 1) /
            threadPerBlock.y,
        (static_cast<decltype(dim3::x)>(vdbInfo.voxPerAtlas.z) + threadPerBlock.z - 1) /
            threadPerBlock.z};
    parallelExec3D<<<blockPerGrid, threadPerBlock>>>(krnlFn, vdbInfo.voxPerAtlas);
}
