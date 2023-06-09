#ifndef KOUEK_DPBXVDB_H
#define KOUEK_DPBXVDB_H

#include <array>
#include <bitset>
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
    enum class State { NeedRebuild, KeepOldAtlas, End };
    std::bitset<static_cast<size_t>(State::End)> states;
    bool testState(State state) const { return states.test(static_cast<size_t>(state)); }
    void setState(State state, bool val = true) { states.set(static_cast<size_t>(state), val); }

    DownsamplePolicy downsamplePolicy = DownsamplePolicy::Avg;
    IDTy rootID = UndefID;
    std::array<CoordValTy, MaxLevNum> chNums{0, 0, 0};
    std::array<CoordTy, MaxLevNum> coverVoxes;

    VDBInfo info;
    VDBDeviceData devDat;

    cudaArray_t atlasArr = nullptr;
    cudaResourceDesc atlasResDesc;
    cudaTextureObject_t atlasTex;
    cudaTextureObject_t atlasDepTex = 0;
    cudaSurfaceObject_t atlasSurf;

    std::queue<IDTy> availableAtlasBrickIDs;

    std::vector<AtlasMap> atlasMaps;
    std::vector<IDTy> chListPool;
    std::array<std::vector<Node>, MaxLevNum> nodePools;

    thrust::device_vector<AtlasMap> d_atlasMaps;
    thrust::device_vector<IDTy> d_chListPool;
    std::array<thrust::device_vector<Node>, MaxLevNum> d_nodePools;

  public:
    Tree() {
        memset(&atlasResDesc, 0, sizeof(atlasResDesc));
        atlasResDesc.resType = cudaResourceTypeArray;
    }
    const bool IsReady() const { return !testState(State::NeedRebuild); }
    const auto &GetInfo() const { return info; }
    const auto &GetDeviceData() const { return devDat; }

    void Configure(const std::array<uint8_t, MaxLevNum> &log2Dims, uint8_t apronWidth = 1,
                   bool useDPBX = false,
                   DownsamplePolicy downsamplePolicy = DownsamplePolicy::Max) {
        if (useDPBX)
            ++apronWidth; // to avoid tri-linear filter with depth vox, add 1 layer of arpon

        this->downsamplePolicy = downsamplePolicy;

        for (uint8_t lev = 0; lev < MaxLevNum; ++lev) {
            info.log2Dims[lev] = log2Dims[lev];
            info.dims[lev] = static_cast<CoordValTy>(1) << log2Dims[lev];
            chNums[lev] = static_cast<CoordValTy>(info.dims[lev]) * info.dims[lev] * info.dims[lev];
            coverVoxes[lev] =
                lev > 0 ? info.dims[lev] * coverVoxes[lev - 1] : CoordTy{info.dims[lev]};
            info.vDlts[lev] = glm::vec3(coverVoxes[lev]) / glm::vec3(info.dims[lev]);
        }

        info.useDPBX = useDPBX;
        info.apronWidth = apronWidth;
        info.voxPerAtlasBrick =
            info.dims[0] + ((static_cast<CoordValTy>(apronWidth) + (useDPBX ? 1 : 0)) << 1);
        info.apronWidAndDep = apronWidth + (useDPBX ? 1 : 0);
        info.minDepIdx = -static_cast<CoordValTy>(info.apronWidAndDep);
        info.maxDepIdx = info.dims[0] - 1 + info.apronWidAndDep;

        setState(State::NeedRebuild);
    }

    void RebuildAsDense(const std::vector<float> &src, const CoordTy &oldVoxPerVol,
                        const AxisTransform &axisTr = {0, 1, 2}) {
        info.thresh = 0.f;
        info.voxPerVol = axisTr.TransformDimension(oldVoxPerVol);
        info.brickPerVol = {(info.voxPerVol.x + info.dims[0] - 1) / info.dims[0],
                            (info.voxPerVol.y + info.dims[0] - 1) / info.dims[0],
                            (info.voxPerVol.z + info.dims[0] - 1) / info.dims[0]};

        clearHostData();

        auto d_src = loadByAxisTransform(src, oldVoxPerVol, axisTr);
        {
            for (CoordValTy z = 0; z < info.brickPerVol.z; ++z)
                for (CoordValTy y = 0; y < info.brickPerVol.y; ++y)
                    for (CoordValTy x = 0; x < info.brickPerVol.x; ++x) {
                        CoordTy pos{x, y, z};
                        pos *= info.dims[0];
                        activateSpace(rootID, pos, UndefID, 0);
                    }
        }

        setState(State::KeepOldAtlas, false);
        updateAtlas(d_src);
        setState(State::NeedRebuild, false);
    }

    void RebuildAsSparse(const std::vector<float> &src, const CoordTy &oldVoxPerVol,
                         float threshold, const AxisTransform &axisTr = {0, 1, 2}) {
        info.thresh = threshold;
        info.voxPerVol = axisTr.TransformDimension(oldVoxPerVol);
        info.brickPerVol = {(info.voxPerVol.x + info.dims[0] - 1) / info.dims[0],
                            (info.voxPerVol.y + info.dims[0] - 1) / info.dims[0],
                            (info.voxPerVol.z + info.dims[0] - 1) / info.dims[0]};

        clearHostData();

        auto d_src = loadByAxisTransform(src, oldVoxPerVol, axisTr);
        {
            auto downsampled = downsample(d_src, downsamplePolicy, info);
            for (CoordValTy z = 0; z < info.brickPerVol.z; ++z)
                for (CoordValTy y = 0; y < info.brickPerVol.y; ++y)
                    for (CoordValTy x = 0; x < info.brickPerVol.x; ++x)
                        if (downsampled[(z * info.brickPerVol.y + y) * info.brickPerVol.x + x] >=
                            info.thresh) {
                            CoordTy pos{x, y, z};
                            pos *= info.dims[0];
                            activateSpace(rootID, pos, UndefID, 0);
                        }
        }

        setState(State::KeepOldAtlas, false);
        updateAtlas(d_src);
        setState(State::NeedRebuild, false);
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
        devDat.atlasDepTex = atlasDepTex;
    }

  private:
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
    IDTy appendNodeAtPool(uint8_t lev, const CoordTy &idx3) {
        auto &pool = nodePools[lev];
        IDTy idx = pool.size();
        auto &node = pool.emplace_back();
        node.Init(lev, idx3);

        return Node::GetID(lev, idx);
    }
    Node &getNode(IDTy nodeID) { return nodePools[Node::ID2Lev(nodeID)][Node::ID2Idx(nodeID)]; }

  private:
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

  private:
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

  private:
    void reserveAtlas(const CoordTy &voxPerAtlasNew) {
        if (atlasResDesc.res.array.array) {
            CUDACheck(cudaDestroyTextureObject(atlasTex));
            CUDACheck(cudaDestroySurfaceObject(atlasSurf));
            if (atlasDepTex != 0)
                CUDACheck(cudaDestroyTextureObject(atlasDepTex));
            atlasResDesc.res.array.array = nullptr;
        }

        if (atlasArr &&
            (info.voxPerAtlas.x != voxPerAtlasNew.x || info.voxPerAtlas.y != voxPerAtlasNew.y)) {
            // Atlas should only increase on Z-axis
            CUDACheck(cudaFreeArray(atlasArr));
            atlasArr = nullptr;
        }

        auto oldArr = atlasArr;
        auto availableLayerStart = oldArr && testState(State::KeepOldAtlas)
                                       ? info.brickPerAtlas.z
                                       : static_cast<CoordValTy>(0);

        cudaMemcpy3DParms cp;
        memset(&cp, 0, sizeof(cp));
        cp.kind = cudaMemcpyDeviceToDevice;
        cp.srcArray = oldArr;
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
        if (oldArr) {
            if (testState(State::KeepOldAtlas)) {
                cp.dstArray = atlasArr;
                CUDACheck(cudaMemcpy3D(&cp));
            }
            CUDACheck(cudaFreeArray(oldArr));
        }

        for (auto z = availableLayerStart; z < info.brickPerAtlas.z; ++z)
            for (CoordValTy y = 0; y < info.brickPerAtlas.y; ++y)
                for (CoordValTy x = 0; x < info.brickPerAtlas.x; ++x)
                    availableAtlasBrickIDs.emplace(info.AtlastBrickIdx32ID({x, y, z}));

        info.brickNumPerAtlas =
            static_cast<IDTy>(info.brickPerAtlas.x) * info.brickPerAtlas.y * info.brickPerAtlas.z;
        atlasMaps.resize(info.brickNumPerAtlas, UndefAtlasMap);

        setState(State::KeepOldAtlas);
    }
    void allocAtlasBrick(IDTy leafID) {
        if (availableAtlasBrickIDs.empty()) {
            auto voxPerAtlasNew = info.voxPerAtlas;
            voxPerAtlasNew.z += info.voxPerAtlasBrick;
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
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.borderColor[0] = texDesc.borderColor[1] = texDesc.borderColor[2] =
            texDesc.borderColor[3] = 0.f;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        CUDACheck(cudaCreateTextureObject(&atlasTex, &atlasResDesc, &texDesc, nullptr));

        texDesc.filterMode = cudaFilterModePoint;
        CUDACheck(cudaCreateTextureObject(&atlasDepTex, &atlasResDesc, &texDesc, nullptr));
    }

  private:
    void clearHostData() {
        for (auto &pool : nodePools)
            pool.clear();
        chListPool.clear();
        atlasMaps.clear();
        availableAtlasBrickIDs = decltype(availableAtlasBrickIDs)();

        rootID = UndefID;
    }

    void updateAtlas(const thrust::device_vector<float> &d_src) {
        auto voxPerAtlasNew = [&]() {
            CoordTy ret{info.dims[0] + (static_cast<CoordValTy>(info.apronWidAndDep) << 1)};
            auto brickPerAtlas = info.brickPerVol;
            brickPerAtlas.z >>= 1;
            ret *= brickPerAtlas;
            return ret;
        }();
        reserveAtlas(voxPerAtlasNew);

        auto leafBrickNum = nodePools[0].size();
        for (IDTy idx = 0; idx < leafBrickNum; ++idx)
            allocAtlasBrick(Node::GetID(0, idx));

        setupAtlasAccess();
        uploadDeviceData();
        resample(d_src, info, devDat);
        resampleDepth(info, devDat);
    }
};

} // namespace dpbxvdb

#endif // !KOUEK_DPBXVDB_H
