#include "resample.cuh"

#include <glm/gtc/matrix_transform.hpp>

thrust::device_vector<float> dpbxvdb::loadByAxisTransform(const std::vector<float> &src,
                                                          const CoordTy &oldVoxPerVol,
                                                          const AxisTransform &axisTr) {
    if (axisTr.IsNatural())
        return thrust::device_vector<float>(src);

    thrust::device_vector<float> tmp(src);
    thrust::device_vector<float> ret(src.size());
    auto krnlFn = [dst = thrust::raw_pointer_cast(ret.data()),
                   src = thrust::raw_pointer_cast(tmp.data()), oldVoxPerVol, axisTr,
                   dimYX = static_cast<size_t>(
                       oldVoxPerVol.y * oldVoxPerVol.x)] __device__(CoordTy oldIdx3, IDTy oldIdx) {
        CoordTy idx3;
        for (uint8_t xyz = 0; xyz < 3; ++xyz) {
            auto trAxis = axisTr[xyz] % 3;
            if (axisTr[xyz] / 3 != 0)
                idx3[xyz] = oldVoxPerVol[trAxis] - 1 - oldIdx3[trAxis];
            else
                idx3[xyz] = oldIdx3[trAxis];
        }

        auto idx = idx3.z * dimYX + idx3.y * oldVoxPerVol.x + idx3.x;
        dst[idx] = src[oldIdx];
    };
    dim3 threadPerBlock{8, 8, 8};
    dim3 blockPerGrid{
        (static_cast<decltype(dim3::x)>(oldVoxPerVol.x) + threadPerBlock.x - 1) / threadPerBlock.x,
        (static_cast<decltype(dim3::x)>(oldVoxPerVol.y) + threadPerBlock.y - 1) / threadPerBlock.y,
        (static_cast<decltype(dim3::x)>(oldVoxPerVol.z) + threadPerBlock.z - 1) / threadPerBlock.z};
    parallelExec3D<<<blockPerGrid, threadPerBlock>>>(krnlFn, oldVoxPerVol);

    return ret;
}

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

        auto nodeID = vdbDat.atlasMaps[vdbInfo.AtlastBrickIdx32ID(brickIdx3)].node;
        if (nodeID == UndefID)
            return;

        auto vMin = vdbDat.nodePools[0][Node::ID2Idx(nodeID)].idx3;
        vIdx3 =
            glm::clamp(vMin + vIdx3, glm::zero<CoordTy>(),
                       vdbInfo.voxPerVol - static_cast<CoordValTy>(1)); // vox idx3 in Volume Space

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

void dpbxvdb::resampleDepth(const VDBInfo &vdbInfo, const VDBDeviceData &vdbDat) {
    if (!vdbInfo.useDPBX)
        return;

    auto apronWidAndDep = static_cast<CoordValTy>(vdbInfo.apronWidAndDep);
    auto voxPerBrick = vdbInfo.dims[0];
    auto minDepIdx = vdbInfo.minDepIdx;
    auto maxDepIdx = vdbInfo.maxDepIdx;
    auto krnlFn = [vdbInfo, vdbDat, apronWidAndDep, voxPerBrick, minDepIdx,
                   maxDepIdx] __device__(CoordTy aIdx3, IDTy aIdx) {
        CoordTy brickIdx3{aIdx3.x / vdbInfo.voxPerAtlasBrick, aIdx3.y / vdbInfo.voxPerAtlasBrick,
                          aIdx3.z / vdbInfo.voxPerAtlasBrick};
        auto aMin = vdbInfo.voxPerAtlasBrick * brickIdx3 + apronWidAndDep;
        aIdx3 -= aMin; // vox idx3 in Brick Space

        auto depSign = glm::zero<CoordTy>();
#pragma unroll
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (aIdx3[xyz] == minDepIdx)
                depSign[xyz] = 1;
            else if (aIdx3[xyz] == maxDepIdx)
                depSign[xyz] = -1;

        auto nodeID = vdbDat.atlasMaps[vdbInfo.AtlastBrickIdx32ID(brickIdx3)].node;
        if (nodeID == UndefID)
            return;

        if ((depSign.x | depSign.y | depSign.z) == 0)
            return;

        aIdx3 += aMin;
        if ((depSign.x & depSign.y) || (depSign.y & depSign.z) || (depSign.z & depSign.x)) {
            surf3Dwrite(0.f, vdbDat.atlasSurf, sizeof(float) * aIdx3.x, aIdx3.y, aIdx3.z);
            return;
        }

        auto pos = aIdx3 + apronWidAndDep * depSign;
        CoordValTy t = 0;
        while (true) {
            float val;
            surf3Dread(&val, vdbDat.atlasSurf, sizeof(float) * pos.x, pos.y, pos.z);
            if (val >= vdbInfo.thresh)
                break;
            pos += depSign;
            ++t;
            if (t >= voxPerBrick)
                break;
        }
        surf3Dwrite(static_cast<float>(t), vdbDat.atlasSurf, sizeof(float) * aIdx3.x, aIdx3.y,
                    aIdx3.z);
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
