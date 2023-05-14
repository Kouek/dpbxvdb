#ifndef KOUEK_MAIN_H
#define KOUEK_MAIN_H

#include <array>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <dpbxvdb/dpbxvdb.h>

#include <cuda_runtime.h>

using DpbxRawVoxTy = uint8_t;

struct RenderParam {
    GLuint texID;
    glm::uvec2 res;
    glm::mat4 proj;
    glm::vec3 bkgrndCol;
    float dt;
};

enum class RenderTarget { Vol, BrickL0, BrickL1, BrickL2, Depth, SkipTimeDlt, End };
constexpr std::array RenderTargetNames{"Volume",   "Brick L0", "Brick L1",
                                       "Brick L2", "Depth",    "Skip Time Delta"};

void release();
void setRenderParam(const RenderParam &param);
void setDPBXParam(const dpbxvdb::VDBInfo &vdbInfo, const dpbxvdb::VDBDeviceData &vdbDat);
void setTF(const std::vector<float> &flatTF);
void render(const glm::vec3 &camPos, const glm::mat3 &camRot, RenderTarget rndrTarget);

#endif // !KOUEK_MAIN_H
