#ifndef KOUEK_MAIN_H
#define KOUEK_MAIN_H

#include <array>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <dpbxvdb/dpbxvdb.h>

#include <cuda_runtime.h>

#define ENABLE_PERFORMANCE_TEST

using DpbxRawVoxTy = uint8_t;

struct RenderParam {
    bool usePhongShading;
    GLuint texID;
    glm::uvec2 res;
    glm::vec3 bkgrndCol;
    glm::vec3 lightPos;
    glm::vec3 lightCol;
    float dt;
    float ka, kd, ks, shininess;
};

enum class RenderTarget { Vol, BrickL0, BrickL1, BrickL2, Depth, SkipTimeDiff, End };
constexpr std::array RenderTargetNames{"Volume",   "Brick L0", "Brick L1",
                                       "Brick L2", "Depth",    "Skip Time Diff"};

void release();
void setRenderParam(const RenderParam &param);
void setDPBXParam(const dpbxvdb::VDBInfo &vdbInfo, const dpbxvdb::VDBDeviceData &vdbDat);
void setTF(const std::vector<float> &flatTF);
void render(const glm::mat4 &unProj, const glm::mat4 &tr, RenderTarget rndrTarget, float &costInMs);

#endif // !KOUEK_MAIN_H
