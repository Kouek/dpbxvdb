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
    float thresh;
    float dt;
};

enum class RenderTarget { DenseVol = 0, SparseVol, TileL0, TileL1, TileL2, End };

constexpr std::array RenderTargetNames{"DenseVol", "SparseVol", "TileL0", "TileL1", "TileL2"};

void release();
void setRenderParam(const RenderParam &param);
void setVolume();
void setTF(const std::vector<float> &flatTF);
void render(const glm::vec3 &camPos, const glm::mat3 &camRot, RenderTarget rndrTarget,
            const dpbxvdb::Tree &vdb);

#endif // !KOUEK_MAIN_H
