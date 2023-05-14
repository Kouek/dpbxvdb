#ifndef KOUEK_RAW_VOLUME_LOADER_H
#define KOUEK_RAW_VOLUME_LOADER_H

#include <algorithm>
#include <array>
#include <fstream>
#include <numeric>
#include <vector>

#include <dpbxvdb/dpbxvdb.h>

#include <glm/glm.hpp>

template <typename RawVoxTy> class RawVolume {
  private:
    dpbxvdb::Tree vdb;

  public:
    const auto &GetVDB() const { return vdb; }
    void LoadAsDense(const std::string &path, const glm::uvec3 &dim, bool useDPBX,
                     const std::array<uint8_t, 3> &log2Dims = {4, 4, 5}) {
        auto src = loadSrc(path, dim);
        vdb.Configure(log2Dims, 1, useDPBX);
        vdb.RebuildAsDense(src, dim);
    }
    void LoadAsSparse(const std::string &path, const glm::uvec3 &dim, const RawVoxTy &threshold,
                      bool useDPBX, const std::array<uint8_t, 3> &log2Dims = {4, 4, 5}) {
        auto src = loadSrc(path, dim);
        vdb.Configure(log2Dims, 1, useDPBX);
        vdb.RebuildAsSparse(src, dim, rawVoxTy2Float(threshold));
    }

  private:
    static inline float rawVoxTy2Float(RawVoxTy rawVal) {
        return static_cast<float>(rawVal) / std::numeric_limits<RawVoxTy>::max();
    }

    std::vector<float> loadSrc(const std::string &path, const glm::uvec3 &dim) {
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
        std::transform(rawDat.begin(), rawDat.end(), dat.begin(),
                       [](const RawVoxTy &rawVal) { return rawVoxTy2Float(rawVal); });

        return dat;
    }
};

#endif // !KOUEK_RAW_VOLUME_LOADER_H
