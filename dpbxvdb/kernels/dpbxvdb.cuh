#ifndef KOUEK_DPBXVDB_CUH
#define KOUEK_DPBXVDB_CUH

#include <iostream>

#include <cuda.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>

#if defined(__CUDACC__) || defined(__HIP__)
// Only define __hostdev__ when using NVIDIA CUDA or HIP compiler
#define __dpbxvdb_hostdev__ __host__ __device__
#else
#define __dpbxvdb_hostdev__
#endif

#define dpbxvdb_align alignas(16)

namespace dpbxvdb {

using IDTy = uint64_t;
using ChildIdxTy = uint32_t;
using CoordValTy = int64_t;
using CoordTy = glm::vec<3, CoordValTy>;

constexpr auto UndefID = std::numeric_limits<IDTy>::max();
constexpr auto UndefChIdx = std::numeric_limits<ChildIdxTy>::max();

constexpr auto Epsilon = .001f;

constexpr uint8_t MaxLevNum = 3;

template <typename F> void __global__ parallelExec3D(F f, CoordTy dim) {
    CoordTy idx3{blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y,
                 blockIdx.z * blockDim.z + threadIdx.z};
    auto idx = (static_cast<IDTy>(idx3.z) * dim.y + idx3.y) * dim.x + idx3.x;
    if (idx3.x >= dim.x || idx3.y >= dim.y || idx3.z >= dim.z)
        return;

    f(idx3, idx);
}

inline void processCUDAError(cudaError_t err, const char *file, int line) {
    if (err == cudaSuccess)
        return;
    std::cerr << "CUDA error: " << cudaGetErrorName(err) << " happened at line: " << line
              << " in file: " << file << std::endl;
}
#define CUDACheck(f) dpbxvdb::processCUDAError(f, __FILE__, __LINE__)
#define CUDACheckLast dpbxvdb::processCUDAError(cudaGetLastError(), __FILE__, __LINE__)

struct dpbxvdb_align Node {
    uint8_t lev;
    CoordTy idx3;
    CoordTy atlasIdx3;
    IDTy chList;
    IDTy par;

    void Init(uint8_t lev, const CoordTy &idx3) {
        this->lev = lev;
        this->idx3 = idx3;
        chList = par = UndefID;
    }

    static __dpbxvdb_hostdev__ IDTy ID2Lev(IDTy id) { return id & 0xff; }
    static __dpbxvdb_hostdev__ IDTy ID2Idx(IDTy id) { return id >> 8; }
    static __dpbxvdb_hostdev__ IDTy GetID(uint8_t lev, IDTy idx) { return lev | (idx << 8); }
};

struct dpbxvdb_align AtlasMap {
    IDTy node;
    CoordTy atlasBrickIdx3;
};

constexpr AtlasMap UndefAtlasMap{UndefID};

struct dpbxvdb_align VDBInfo {
    uint8_t apronWidth;
    uint8_t apronWidAndDep;
    uint8_t topLev;
    uint8_t log2Dims[MaxLevNum];
    bool useDPBX;
    CoordValTy dims[MaxLevNum];
    CoordValTy voxPerAtlasBrick;
    CoordValTy minDepIdx;
    CoordValTy maxDepIdx;
    IDTy brickNumPerAtlas;
    CoordTy voxPerVol;
    CoordTy voxPerAtlas;
    CoordTy brickPerVol;
    CoordTy brickPerAtlas;
    glm::vec3 vDlts[MaxLevNum];
    float thresh;

    __dpbxvdb_hostdev__ ChildIdxTy PosInBrick2ChildIdx(const CoordTy &pos, uint8_t lev) const {
        return (((pos.z << log2Dims[lev]) + pos.y) << log2Dims[lev]) + pos.x;
    }
    __dpbxvdb_hostdev__ CoordTy ChildIndex2PosInNode(ChildIdxTy chIdx, uint8_t lev) const {
        CoordTy ret;
        ChildIdxTy mask = static_cast<ChildIdxTy>(dims[lev]) - 1;
        ret.z = (chIdx & (mask << (log2Dims[lev] << 1))) >> (log2Dims[lev] << 1);
        ret.y = (chIdx & (mask << log2Dims[lev])) >> log2Dims[lev];
        ret.x = chIdx & mask;
        return ret;
    }
    __dpbxvdb_hostdev__ IDTy AtlastBrickIdx32ID(const CoordTy &idx3) const {
        return (static_cast<IDTy>(idx3.z) * brickPerAtlas.y + idx3.y) * brickPerAtlas.x + idx3.x;
    }
    __dpbxvdb_hostdev__ CoordTy AtlasBrickID2Idx3(IDTy id) const {
        auto z = id / (static_cast<IDTy>(brickPerAtlas.x) * brickPerAtlas.y);
        auto xyz = z * brickPerAtlas.x * brickPerAtlas.y;
        auto y = (id - xyz) / brickPerAtlas.x;
        auto xy = y * brickPerAtlas.x;
        return CoordTy{id - xy - xyz, y, z};
    }
};

struct dpbxvdb_align VDBDeviceData {
    AtlasMap *atlasMaps;
    Node *nodePools[MaxLevNum];
    IDTy *chListPool;

    cudaSurfaceObject_t atlasSurf;
    cudaTextureObject_t atlasTex;
    cudaTextureObject_t atlasDepTex;
};

struct HDDA3D {
    glm::ivec3 sign; // signs of ray dir
    glm::ivec3 mask; // 0 for should NOT and 1 for should move on XYZ axis
    CoordTy chIdx3;  // index of child node relative to parent node
    glm::vec3 t;     // (current time, next time, hit status)
    glm::vec3 tSide; // time that ray intersects with next plane in XYZ direction
    glm::vec3 pos;   // ray pos in Index Space
    glm::vec3 dir;   // ray dir in Index Space
    glm::vec3 tDlt;  // time delta

    __dpbxvdb_hostdev__ void Init(const glm::vec3 &rayPos, const glm::vec3 &rayDir,
                                  const glm::vec3 &t) {
        pos = rayPos;
        dir = rayDir;
        sign = {dir.x > 0.f   ? 1
                : dir.x < 0.f ? -1
                              : 0,
                dir.y > 0.f   ? 1
                : dir.y < 0.f ? -1
                              : 0,
                dir.z > 0.f   ? 1
                : dir.z < 0.f ? -1
                              : 0};
        this->t = t;
    }

    __dpbxvdb_hostdev__ void Prepare(const glm::vec3 &vMin, const glm::vec3 &vDlt) {
        tDlt = glm::abs(vDlt / dir);
        auto pFlt = (pos + t.x * dir - vMin) / vDlt;
        tSide = ((glm::floor(pFlt) - pFlt + .5f) * glm::vec3{sign} + .5f) * tDlt + t.x;
        chIdx3 = glm::floor(pFlt);
    }

    __dpbxvdb_hostdev__ void Next() {
        using GLMIntTy = decltype(glm::ivec3::x);
        mask.x = static_cast<GLMIntTy>((tSide.x < tSide.y) & (tSide.x <= tSide.z));
        mask.y = static_cast<GLMIntTy>((tSide.y < tSide.z) & (tSide.y <= tSide.x));
        mask.z = static_cast<GLMIntTy>((tSide.z < tSide.x) & (tSide.z <= tSide.y));
        t.y = mask.x ? tSide.x : mask.y ? tSide.y : mask.z ? tSide.z : INFINITY;
    }

    __dpbxvdb_hostdev__ void Step() {
        t.x = t.y;
        tSide.x = isinf(tDlt.x) ? INFINITY : mask.x ? tSide.x + tDlt.x : tSide.x;
        tSide.y = isinf(tDlt.y) ? INFINITY : mask.y ? tSide.y + tDlt.y : tSide.y;
        tSide.z = isinf(tDlt.z) ? INFINITY : mask.z ? tSide.z + tDlt.z : tSide.z;
        chIdx3 += mask * sign;
    }
};

struct DPBXDDA2D {
    float t;
    float dep;
    glm::ivec3 depSign;
    glm::ivec3 sign;
    glm::ivec3 mask;
    glm::vec3 tSide;
    glm::vec3 tDlt;
    glm::vec3 pos;

    __dpbxvdb_hostdev__ bool Init(const glm::vec3 &rayPos, const glm::vec3 &rayDir, float t,
                                  const glm::vec3 &posInBrick, const VDBInfo &vdbInfo) {
        constexpr auto Sqrt2Div2 = 0.70710678f;

        dep = 0.f;
        pos = posInBrick;
        sign = {rayDir.x > 0.f   ? 1
                : rayDir.x < 0.f ? -1
                                 : 0,
                rayDir.y > 0.f   ? 1
                : rayDir.y < 0.f ? -1
                                 : 0,
                rayDir.z > 0.f   ? 1
                : rayDir.z < 0.f ? -1
                                 : 0};
        this->t = t;

        {
            float max = vdbInfo.dims[0] - 1;
            glm::vec3 distToAxis{sign.x == 0  ? INFINITY
                                 : sign.x > 0 ? pos.x
                                              : max - pos.x,
                                 sign.y == 0  ? INFINITY
                                 : sign.y > 0 ? pos.y
                                              : max - pos.y,
                                 sign.z == 0  ? INFINITY
                                 : sign.z > 0 ? pos.z
                                              : max - pos.z};
            depSign.x = (distToAxis.x <= distToAxis.y && distToAxis.x <= distToAxis.z) ? sign.x : 0;
            depSign.y = (distToAxis.y <= distToAxis.z && distToAxis.y <= distToAxis.x) ? sign.y : 0;
            depSign.z = (distToAxis.z <= distToAxis.x && distToAxis.z <= distToAxis.y) ? sign.z : 0;
        }

        tDlt = glm::abs(vdbInfo.vDlts[0] / rayDir);
        auto pFlt = pos / vdbInfo.vDlts[0];
        tSide = ((glm::floor(pFlt) - pFlt + .5f) * glm::vec3{sign} + .5f) * tDlt + t;

        if (depSign.x != 0) {
            pos.x = depSign.x == 1 ? vdbInfo.minDepIdx : vdbInfo.maxDepIdx;
            if (glm::abs(rayDir.x) < Sqrt2Div2)
                return false;
        }
        if (depSign.y != 0) {
            pos.y = depSign.y == 1 ? vdbInfo.minDepIdx : vdbInfo.maxDepIdx;
            if (glm::abs(rayDir.y) < Sqrt2Div2)
                return false;
        }
        if (depSign.z != 0) {
            pos.z = depSign.z == 1 ? vdbInfo.minDepIdx : vdbInfo.maxDepIdx;
            if (glm::abs(rayDir.z) < Sqrt2Div2)
                return false;
        }

        return depSign.x | depSign.y | depSign.z;
    }

    __dpbxvdb_hostdev__ void StepNext() {
        using GLMIntTy = decltype(glm::ivec3::x);
        mask.x = static_cast<GLMIntTy>((tSide.x < tSide.y) & (tSide.x <= tSide.z));
        mask.y = static_cast<GLMIntTy>((tSide.y < tSide.z) & (tSide.y <= tSide.x));
        mask.z = static_cast<GLMIntTy>((tSide.z < tSide.x) & (tSide.z <= tSide.y));

        t = mask.x ? tSide.x : mask.y ? tSide.y : mask.z ? tSide.z : INFINITY;

        tSide.x = isinf(tDlt.x) ? INFINITY : mask.x ? tSide.x + tDlt.x : tSide.x;
        tSide.y = isinf(tDlt.y) ? INFINITY : mask.y ? tSide.y + tDlt.y : tSide.y;
        tSide.z = isinf(tDlt.z) ? INFINITY : mask.z ? tSide.z + tDlt.z : tSide.z;

        if (depSign.x != 0 && mask.x != 0) {
            dep += 1.f;
            mask.x = 0;
        }
        if (depSign.y != 0 && mask.y != 0) {
            dep += 1.f;
            mask.y = 0;
        }
        if (depSign.z != 0 && mask.z != 0) {
            dep += 1.f;
            mask.z = 0;
        }
        pos += mask * sign;
    }
};

inline __dpbxvdb_hostdev__ glm::vec3 rayIntersectAABB(const glm::vec3 &rayPos,
                                                      const glm::vec3 &rayDir,
                                                      const glm::vec3 &AABBMin,
                                                      const glm::vec3 &AABBMax) {
    register float ht[8];
    ht[0] = (AABBMin.x - rayPos.x) / rayDir.x;
    ht[1] = (AABBMax.x - rayPos.x) / rayDir.x;
    ht[2] = (AABBMin.y - rayPos.y) / rayDir.y;
    ht[3] = (AABBMax.y - rayPos.y) / rayDir.y;
    ht[4] = (AABBMin.z - rayPos.z) / rayDir.z;
    ht[5] = (AABBMax.z - rayPos.z) / rayDir.z;
    ht[6] =
        glm::max(glm::max(glm::min(ht[0], ht[1]), glm::min(ht[2], ht[3])), glm::min(ht[4], ht[5]));
    ht[7] =
        glm::min(glm::min(glm::max(ht[0], ht[1]), glm::max(ht[2], ht[3])), glm::max(ht[4], ht[5]));
    ht[6] = (ht[6] < 0.f) ? 0.f : ht[6];
    return {ht[6], ht[7], (ht[7] < ht[6] || ht[7] < 0.f) ? -1.f : 0.f};
}

inline __dpbxvdb_hostdev__ IDTy GetChildID(const VDBDeviceData &vdbDat, const Node &par,
                                           ChildIdxTy chIdx) {
    return vdbDat.chListPool[par.chList + chIdx];
}

inline __dpbxvdb_hostdev__ bool IsChildOn(const VDBDeviceData &vdbDat, const Node &par,
                                          ChildIdxTy chIdx) {
    if (par.chList == UndefID)
        return UndefID;
    return GetChildID(vdbDat, par, chIdx) != UndefID;
}

} // namespace dpbxvdb

#endif // !KOUEK_DPBXVDB_CUH
