#include "resample.cuh"

#include <glm/gtc/matrix_transform.hpp>

void dpbxvdb::resample(const thrust::device_vector<float> &d_src, cudaSurfaceObject_t dstSurf,
                       float threshold, const VDBInfo &vdbInfo, const VDBDeviceData &vdbDat) {
    auto apronWidth = static_cast<CoordValTy>(vdbInfo.apronWidth);
    auto apronAndDepWid = static_cast<CoordValTy>(vdbInfo.apronWidAndDep);
    auto voxPerBrick = vdbInfo.dims[0];
    auto minDepIdx = -1 - vdbInfo.apronWidth;
    auto maxDepIdx = vdbInfo.voxPerAtlasBrick + vdbInfo.apronWidth;
    auto krnlFn = [src = thrust::raw_pointer_cast(d_src.data()), dstSurf, threshold, vdbInfo,
                   vdbDat, apronWidth, apronAndDepWid, voxPerBrick, minDepIdx,
                   maxDepIdx] __device__(CoordTy aIdx3, IDTy aIdx) {
        CoordTy brickIdx3{aIdx3.x / vdbInfo.voxPerAtlasBrick, aIdx3.y / vdbInfo.voxPerAtlasBrick,
                          aIdx3.z / vdbInfo.voxPerAtlasBrick};
        auto aMin = vdbInfo.voxPerAtlasBrick * brickIdx3 + apronAndDepWid;
        auto vIdx3 = aIdx3 - aMin; // vox idx3 in Brick Space

        uint8_t depDir = 0;
#pragma unroll
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (vIdx3[xyz] == minDepIdx)
                depDir |= static_cast<uint8_t>(1) << (xyz << 1);
            else if (vIdx3[xyz] == maxDepIdx)
                depDir |= static_cast<uint8_t>(1) << ((xyz << 1) + 1);

        auto nodeID = vdbDat.atlasMaps[vdbInfo.AtlastBrickIdx32ID(brickIdx3)].node;
        if (nodeID == UndefID)
            return;

        auto vMin = vdbDat.nodePools[0][Node::ID2Idx(nodeID)].idx3;
        vIdx3 = glm::clamp(vMin + vIdx3, glm::zero<CoordTy>(),
                           vdbInfo.voxPerVol); // vox idx3 in Volume Space
        if (vdbInfo.useDPBX && depDir != 0) {
            CoordTy step{(depDir & 0b000001) != 0   ? 1
                         : (depDir & 0b000010) != 0 ? -1
                                                    : 0,
                         (depDir & 0b000100) != 0   ? 1
                         : (depDir & 0b001000) != 0 ? -1
                                                    : 0,
                         (depDir & 0b010000) != 0   ? 1
                         : (depDir & 0b100000) != 0 ? -1
                                                    : 0};
            auto pos = vIdx3 + apronWidth * step;
            CoordValTy t = 0;
            while (true) {
                auto val = src[(pos.z * vdbInfo.voxPerVol.y + pos.y) * vdbInfo.voxPerVol.x + pos.x];
                if (val >= threshold)
                    break;
                pos += step;
                ++t;
                if (t >= voxPerBrick)
                    break;
            }
            auto dep = static_cast<float>(t) / voxPerBrick;
            surf3Dwrite(dep, dstSurf, sizeof(float) * aIdx3.x, aIdx3.y, aIdx3.z);
            return;
        }

        auto vIdx = (vIdx3.z * vdbInfo.voxPerVol.y + vIdx3.y) * vdbInfo.voxPerVol.x + vIdx3.x;
        auto v = src[vIdx];
        surf3Dwrite(v, dstSurf, sizeof(float) * aIdx3.x, aIdx3.y, aIdx3.z);
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
