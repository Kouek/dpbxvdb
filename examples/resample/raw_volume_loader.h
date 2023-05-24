#ifndef KOUEK_RAW_VOLUME_LOADER_H
#define KOUEK_RAW_VOLUME_LOADER_H

#include <algorithm>
#include <array>
#include <fstream>
#include <numeric>
#include <vector>

#include <dpbxvdb/dpbxvdb.h>

#include <glm/glm.hpp>

using AxisTransform = glm::i8vec3;

template <typename RawVoxTy> class RawVolume {
  private:
    dpbxvdb::Tree vdb;

  public:
    const auto &GetVDB() const { return vdb; }
    void LoadAsDense(const std::string &path, const glm::uvec3 &dim, bool useDPBX,
                     const std::array<uint8_t, 3> &log2Dims = {4, 4, 5},
                     const AxisTransform &axisTransform = {0, 1, 2}) {
        auto src = loadSrc(path, dim, axisTransform);
        auto trDim = trDimByAxisTr(dim, axisTransform);
        vdb.Configure(log2Dims, 1, useDPBX);
        vdb.RebuildAsDense(src, trDim);
    }
    void LoadAsSparse(const std::string &path, const glm::uvec3 &dim, const RawVoxTy &threshold,
                      bool useDPBX, const std::array<uint8_t, 3> &log2Dims = {4, 4, 5},
                      const AxisTransform &axisTransform = {0, 1, 2}) {
        auto src = loadSrc(path, dim, axisTransform);
        auto trDim = trDimByAxisTr(dim, axisTransform);
        vdb.Configure(log2Dims, 1, useDPBX);
        vdb.RebuildAsSparse(src, trDim, rawVoxTy2Float(threshold));
    }

  private:
    static inline float rawVoxTy2Float(RawVoxTy rawVal) {
        return static_cast<float>(rawVal) / std::numeric_limits<RawVoxTy>::max();
    }
    static inline glm::uvec3 trDimByAxisTr(const glm::uvec3& oldDim, const AxisTransform& axisTransform) {
        glm::uvec3 newDim;
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            newDim[xyz] = oldDim[axisTransform[xyz] % 3];
        return newDim;
    }

    std::vector<float> loadSrc(const std::string &path, const glm::uvec3 &dim,
                               const AxisTransform &axisTransform) {
        std::ifstream is(path, std::ios::in | std::ios::binary | std::ios::ate);
        if (!is.is_open())
            throw std::runtime_error("Cannot open file: " + path + " .");

        auto voxNum = static_cast<size_t>(is.tellg()) / sizeof(RawVoxTy);
        {
            auto _voxNum = (size_t)dim.x * dim.y * dim.z;
            if (voxNum < _voxNum)
                throw std::runtime_error("Volume in file: " + path +
                                         " is smaller than the required dim.");
            voxNum = std::min(voxNum, _voxNum);
        }

        std::vector<RawVoxTy> rawDat(voxNum);
        is.seekg(0);
        is.read(reinterpret_cast<char *>(rawDat.data()), sizeof(RawVoxTy) * voxNum);
        is.close();

        std::vector<float> dat(rawDat.size());
        size_t dimYX = dim.y * dim.x;
        for (size_t idxRaw = 0; idxRaw < rawDat.size(); ++idxRaw) {
            glm::uvec3 idx3Raw, idx3;
            idx3Raw.z = idxRaw / dimYX;
            auto zDimYX = idx3Raw.z * dimYX;
            idx3Raw.y = (idxRaw - zDimYX) / dim.x;
            idx3Raw.x = idxRaw - idx3Raw.y * dim.x - zDimYX;

            for (uint8_t xyz = 0; xyz < 3; ++xyz) {
                auto trAxis = axisTransform[xyz] % 3;
                if (axisTransform[xyz] / 3 != 0)
                    idx3[xyz] = dim[trAxis] - 1 - idx3Raw[trAxis];
                else
                    idx3[xyz] = idx3Raw[trAxis];
            }

            auto idx = static_cast<size_t>(idx3.z) * dimYX + idx3.y * dim.x + idx3.x;
            dat[idx] = rawVoxTy2Float(rawDat[idxRaw]);
        }

        return dat;
    }
};

#endif // !KOUEK_RAW_VOLUME_LOADER_H
