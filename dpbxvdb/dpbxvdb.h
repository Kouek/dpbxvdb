#ifndef KOUEK_DPBXVDB_H
#define KOUEK_DPBXVDB_H

#include <array>
#include <queue>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include <glm/gtc/matrix_transform.hpp>

#include "kernels/downsample.cuh"
#include "kernels/dpbxvdb.cuh"
#include "kernels/resample.cuh"

namespace dpbxvdb {

class Tree {
  private:
    VDBInfo info;
    VDBDeviceData devDat;

    DownsamplePolicy downsamplePolicy = DownsamplePolicy::Avg;
    float scnThresh;
    float tfThresh = 0.f;
    IDTy rootID = UndefID;
    size_t maxVoxNumPerAtlas;
    std::array<CoordValTy, MaxLevNum> chNums{0, 0, 0};
    std::array<CoordTy, MaxLevNum> coverVoxes;

    cudaArray_t atlasArr = nullptr;
    cudaResourceDesc atlasResDesc;
    cudaTextureObject_t atlasTex;
    cudaSurfaceObject_t atlasSurf;

    std::queue<IDTy> availableAtlasBrickIDs;

    std::vector<AtlasMap> atlasMaps;
    std::vector<IDTy> chListPool;
    std::array<std::vector<Node>, MaxLevNum> nodePools;

    thrust::device_vector<AtlasMap> d_atlasMaps;
    thrust::device_vector<IDTy> d_chListPool;
    std::array<thrust::device_vector<Node>, MaxLevNum> d_nodePools;

  public:
    Tree() = default;
    const auto &GetInfo() const { return info; }
    const auto &GetDeviceData() const { return devDat; }

    void Configure(const std::array<uint8_t, MaxLevNum> &log2Dims, uint8_t apronWidth = 1,
                   bool useDPBX = false, DownsamplePolicy downsamplePolicy = DownsamplePolicy::Max,
                   size_t maxVoxNumPerAtlas = 1024 * 1024 * 1024) {
        this->downsamplePolicy = downsamplePolicy;

        for (uint8_t lev = 0; lev < MaxLevNum; ++lev) {
            info.log2Dims[lev] = log2Dims[lev];
            info.dims[lev] = static_cast<CoordValTy>(1) << log2Dims[lev];
            chNums[lev] = static_cast<CoordValTy>(info.dims[lev]) * info.dims[lev] * info.dims[lev];
            coverVoxes[lev] =
                lev > 0 ? info.dims[lev] * coverVoxes[lev - 1] : CoordTy{info.dims[lev]};
            info.tDlts[lev] = glm::vec3(coverVoxes[lev]) / glm::vec3(info.dims[lev]);
        }

        this->maxVoxNumPerAtlas = maxVoxNumPerAtlas;
        info.apronWidth = apronWidth;
        info.apronWidAndDep = apronWidth + (useDPBX ? 1 : 0);
        info.useDPBX = useDPBX;
        info.voxPerAtlasBrick =
            info.dims[0] + ((static_cast<CoordValTy>(info.apronWidth) + (useDPBX ? 1 : 0)) << 1);

        memset(&atlasResDesc, 0, sizeof(atlasResDesc));
        atlasResDesc.resType = cudaResourceTypeArray;
    }

    void RebuildAsDense(const std::vector<float> &src, const CoordTy &voxPerVol) {
        scnThresh = 0.f;
        info.voxPerVol = voxPerVol;
        info.brickPerVol = {(voxPerVol.x + info.dims[0] - 1) / info.dims[0],
                            (voxPerVol.y + info.dims[0] - 1) / info.dims[0],
                            (voxPerVol.z + info.dims[0] - 1) / info.dims[0]};
        thrust::device_vector<float> d_src(src);
        {
            for (CoordValTy z = 0; z < info.brickPerVol.z; ++z)
                for (CoordValTy y = 0; y < info.brickPerVol.y; ++y)
                    for (CoordValTy x = 0; x < info.brickPerVol.x; ++x) {
                        CoordTy pos{x, y, z};
                        pos *= info.dims[0];
                        activateSpace(rootID, pos, UndefID, 0);
                    }
        }
        updateAtlas(d_src);
    }

    void RebuildAsSparse(const std::vector<float> &src, const CoordTy &voxPerVol, float threshold) {
        scnThresh = threshold;
        info.voxPerVol = voxPerVol;
        info.brickPerVol = {(voxPerVol.x + info.dims[0] - 1) / info.dims[0],
                            (voxPerVol.y + info.dims[0] - 1) / info.dims[0],
                            (voxPerVol.z + info.dims[0] - 1) / info.dims[0]};
        thrust::device_vector<float> d_src(src);
        {
            auto downsampled = downsample(d_src, downsamplePolicy, info);
            for (CoordValTy z = 0; z < info.brickPerVol.z; ++z)
                for (CoordValTy y = 0; y < info.brickPerVol.y; ++y)
                    for (CoordValTy x = 0; x < info.brickPerVol.x; ++x)
                        if (downsampled[(z * info.brickPerVol.y + y) * info.brickPerVol.x + x] >=
                            scnThresh) {
                            CoordTy pos{x, y, z};
                            pos *= info.dims[0];
                            activateSpace(rootID, pos, UndefID, 0);
                        }
        }

        updateAtlas(d_src);
    }

  private:
    void uploadDeviceData() {
        for (uint8_t lev = 0; lev < MaxLevNum; ++lev) {
            if (nodePools[lev].empty())
                break;
            info.topLev = lev;
            d_nodePools[lev] = nodePools[lev];
            devDat.nodePools[lev] = thrust::raw_pointer_cast(d_nodePools[lev].data());
        }
        d_chListPool = chListPool;
        devDat.chListPool = thrust::raw_pointer_cast(d_chListPool.data());
        d_atlasMaps = atlasMaps;
        devDat.atlasMaps = thrust::raw_pointer_cast(d_atlasMaps.data());
        devDat.atlasSurf = atlasSurf;
        devDat.atlasTex = atlasTex;
    }

    ChildIdxTy getChildIndexInNode(IDTy nodeID, const CoordTy &pos) {
        auto &node = getNode(nodeID);
        auto &cover = coverVoxes[node.lev];
        if (pos.x >= node.idx3.x && pos.y >= node.idx3.y && pos.z >= node.idx3.z) {
            auto posInBrick = (pos - node.idx3) /
                              (node.lev == 0 ? glm::one<CoordTy>() : coverVoxes[node.lev - 1]);
            if (posInBrick.x < cover.x && posInBrick.y < cover.y && posInBrick.z < cover.z)
                return info.PosInBrick2ChildIdx(posInBrick, node.lev);
        }
        return UndefChIdx;
    }
    void clearPools() {
        for (auto &pool : nodePools)
            pool.clear();
        for (auto &pool : d_nodePools)
            pool.clear();
    }
    IDTy appendNodeAtPool(uint8_t lev, const CoordTy &idx3) {
        auto &pool = nodePools[lev];
        IDTy idx = pool.size();
        auto &node = pool.emplace_back();
        node.Init(lev, idx3);

        return Node::GetID(lev, idx);
    }
    Node &getNode(IDTy nodeID) { return nodePools[Node::ID2Lev(nodeID)][Node::ID2Idx(nodeID)]; }

    IDTy getChildID(IDTy parID, ChildIdxTy chIdx) {
        auto &par = getNode(parID);
        return chListPool[par.chList + chIdx];
    }
    bool isChildOn(IDTy parID, ChildIdxTy chIdx) {
        auto &par = getNode(parID);
        if (par.chList == UndefID)
            return false;
        return getChildID(parID, chIdx) != UndefID;
    }
    void insertChild(IDTy parID, IDTy chID, ChildIdxTy chIdx) {
        auto &par = getNode(parID);
        auto &ch = getNode(chID);
        ch.par = parID;
        if (par.chList == UndefID) {
            par.chList = chListPool.size();
            chListPool.resize(chListPool.size() + chNums[par.lev], UndefID);
        }

        chListPool[par.chList + chIdx] = chID;
    }
    IDTy newAndInsertChild(IDTy parID, const CoordTy &parPos, ChildIdxTy chIdx) {
        auto &par = getNode(parID);
        auto cover = par.lev == 0 ? glm::one<CoordTy>() : coverVoxes[par.lev - 1];
        auto pos = parPos + info.ChildIndex2PosInNode(chIdx, par.lev) * cover;
        auto chID = appendNodeAtPool(par.lev - 1, pos);

        insertChild(parID, chID, chIdx);

        return chID;
    }

    CoordTy getNodePosCoverPosAtLev(uint8_t lev, const CoordTy &pos) {
        auto &cover = coverVoxes[lev];
        auto nodePos = pos;
        nodePos = nodePos / cover * cover;

        return nodePos;
    }
    IDTy reparent(IDTy chID, const CoordTy &pos) {
        auto &child = getNode(chID);
        auto lev = child.lev;
        bool covered = false;

        CoordTy parPos;
        while (!covered && lev < MaxLevNum) {
            ++lev;
            parPos = getNodePosCoverPosAtLev(lev, child.idx3);
            auto tmp = getNodePosCoverPosAtLev(lev, pos);
            if (parPos == tmp)
                covered = true;
        }
        if (lev >= MaxLevNum)
            return UndefID;

        auto parID = appendNodeAtPool(lev, parPos);
        activateSpace(parID, child.idx3, chID, 0);
        auto leafID = activateSpace(parID, pos, UndefID, 0);
        rootID = parID;

        return getNode(leafID).par;
    }

    IDTy activateSpace(IDTy parID, const CoordTy &pos, IDTy stopNodeID, uint8_t stopLev) {
        if (rootID == UndefID && parID == rootID) {
            auto rootPos = getNodePosCoverPosAtLev(stopLev, pos);
            rootID = appendNodeAtPool(stopLev, rootPos);
            parID = rootID;
        }

        auto &par = getNode(parID);
        auto chIdx = getChildIndexInNode(parID, pos);
        if (chIdx != UndefChIdx) {
            if (stopNodeID != UndefID) {
                auto &stopNode = getNode(stopNodeID);
                if (pos == stopNode.idx3 && !isChildOn(parID, chIdx)) {
                    insertChild(parID, stopNodeID, chIdx);
                    return stopNodeID;
                }
            }
            if (par.lev == stopLev)
                return parID;

            IDTy chID;
            if (!isChildOn(parID, chIdx))
                chID = newAndInsertChild(parID, par.idx3, chIdx);
            else
                chID = getChildID(parID, chIdx);

            if (Node::ID2Lev(chID) == 0)
                return chID;
            return activateSpace(chID, pos, stopNodeID, stopLev);
        } else {
            auto parParID = par.par;
            if (parParID == UndefID) {
                parParID = reparent(parID, pos);
                if (parParID == UndefID)
                    return UndefID;
            }
            return activateSpace(parParID, pos, stopNodeID, stopLev);
        }
    }

    void reserveAtlas(const CoordTy &voxPerAtlasNew) {
        assert(!atlasArr || (voxPerAtlasNew.x == info.voxPerAtlas.x &&
                             voxPerAtlasNew.y == info.voxPerAtlas.y),
               "Atlas can only increase on Z-axis!");

        if (voxPerAtlasNew.z <= info.voxPerAtlas.z)
            return;

        if (atlasResDesc.res.array.array) {
            CUDACheck(cudaDestroyTextureObject(atlasTex));
            CUDACheck(cudaDestroySurfaceObject(atlasSurf));
            atlasResDesc.res.array.array = nullptr;
        }

        auto oldArr = atlasArr;
        auto availableLayerStart = oldArr ? info.brickPerVol.z : static_cast<CoordValTy>(0);
        cudaMemcpy3DParms cp;
        cp.kind = cudaMemcpyDeviceToDevice;
        cp.srcArray = oldArr;
        cp.dstArray = atlasArr;
        cp.srcPos = cp.dstPos = {0, 0, 0};
        cp.extent.width = info.voxPerAtlas.x;
        cp.extent.height = info.voxPerAtlas.y;
        cp.extent.depth = info.voxPerAtlas.z;

        auto chnlDesc = cudaCreateChannelDesc<float>();
        cudaExtent extent;
        extent.width = info.voxPerAtlas.x = voxPerAtlasNew.x;
        extent.height = info.voxPerAtlas.y = voxPerAtlasNew.y;
        extent.depth = info.voxPerAtlas.z = voxPerAtlasNew.z;
        info.brickPerAtlas.x = info.voxPerAtlas.x / info.voxPerAtlasBrick;
        info.brickPerAtlas.y = info.voxPerAtlas.y / info.voxPerAtlasBrick;
        info.brickPerAtlas.z = info.voxPerAtlas.z / info.voxPerAtlasBrick;

        CUDACheck(cudaMalloc3DArray(&atlasArr, &chnlDesc, extent));
        if (oldArr)
            CUDACheck(cudaMemcpy3D(&cp));

        for (auto z = availableLayerStart; z < info.brickPerVol.z; ++z)
            for (CoordValTy y = 0; y < info.brickPerAtlas.y; ++y)
                for (CoordValTy x = 0; x < info.brickPerAtlas.x; ++x)
                    availableAtlasBrickIDs.emplace(info.AtlastBrickIdx32ID({x, y, z}));

        info.brickNumPerAtlas =
            static_cast<IDTy>(info.brickPerAtlas.x) * info.brickPerAtlas.y * info.brickPerAtlas.z;
        atlasMaps.resize(info.brickNumPerAtlas, UndefAtlasMap);
    }

    void allocAtlasBrick(IDTy leafID) {
        if (availableAtlasBrickIDs.empty()) {
            auto voxPerAtlasNew = info.voxPerAtlas;
            ++voxPerAtlasNew.z;
            reserveAtlas(voxPerAtlasNew);
        }
        auto atlasBrickID = availableAtlasBrickIDs.front();
        availableAtlasBrickIDs.pop();
        auto &atlasMap = atlasMaps[atlasBrickID];
        atlasMap.node = leafID;
        atlasMap.atlasBrickIdx3 = info.AtlasBrickID2Idx3(atlasBrickID);

        getNode(leafID).atlasIdx3 = info.voxPerAtlasBrick * atlasMap.atlasBrickIdx3 +
                                    static_cast<CoordValTy>(info.apronWidAndDep);
    }

    void setupAtlasAccess() {
        if (atlasResDesc.res.array.array)
            return;

        atlasResDesc.res.array.array = atlasArr;
        CUDACheck(cudaCreateSurfaceObject(&atlasSurf, &atlasResDesc));

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        CUDACheck(cudaCreateTextureObject(&atlasTex, &atlasResDesc, &texDesc, nullptr));
    }

    void updateAtlas(const thrust::device_vector<float> &d_src) {
        auto voxPerAtlasNew = [&]() {
            auto voxNumPerAtlasLayer = static_cast<size_t>(info.voxPerVol.x) * info.voxPerVol.y;
            CoordValTy z = maxVoxNumPerAtlas / voxNumPerAtlasLayer;
            CoordTy ret{info.voxPerVol.x, info.voxPerVol.y, std::min(info.voxPerVol.z, z)};
            ret += static_cast<CoordValTy>(info.apronWidAndDep);
            return ret;
        }();
        reserveAtlas(voxPerAtlasNew);

        auto leafBrickNum = nodePools[0].size();
        for (IDTy idx = 0; idx < leafBrickNum; ++idx)
            allocAtlasBrick(Node::GetID(0, idx));

        setupAtlasAccess();
        uploadDeviceData();
        resample(d_src, atlasSurf, std::max(scnThresh, tfThresh), info, devDat);
    }
};

} // namespace dpbxvdb

#endif // !KOUEK_DPBXVDB_H
