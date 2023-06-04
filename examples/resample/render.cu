#include "render.h"

#include <iostream>

#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <thrust/tuple.h>

struct CURenderParam {
    bool usePhongShading;
    glm::uvec2 res;
    glm::vec3 bkgrndCol;
    float thresh;
    float dt;
    float ka, kd, ks, shininess;
};

static constexpr auto GammaCorrCoef = 1.f / 2.2f;

static cudaGraphicsResource_t rndrTexRes = nullptr;
static cudaResourceDesc resDesc;

static CURenderParam rndr;
static __constant__ CURenderParam dc_rndr;

static cudaTextureObject_t tf = 0;
static __constant__ cudaTextureObject_t dc_tf;

static dpbxvdb::VDBInfo vdbInfo;
static __constant__ dpbxvdb::VDBInfo dc_vdbInfo;
static __constant__ dpbxvdb::VDBDeviceData dc_vdbDat;

void release() {
    if (rndrTexRes != nullptr) {
        CUDACheck(cudaGraphicsUnregisterResource(rndrTexRes));
        rndrTexRes = nullptr;
    }
}

void setRenderParam(const RenderParam &param) {
    rndr.res = param.res;
    rndr.bkgrndCol = param.bkgrndCol;
    rndr.usePhongShading = param.usePhongShading;
    rndr.dt = param.dt;
    rndr.ka = param.ka;
    rndr.kd = param.kd;
    rndr.ks = param.ks;
    rndr.shininess = param.shininess;
    CUDACheck(cudaMemcpyToSymbol(dc_rndr, &rndr, sizeof(rndr)));

    if (rndrTexRes != nullptr)
        CUDACheck(cudaGraphicsUnregisterResource(rndrTexRes));
    CUDACheck(cudaGraphicsGLRegisterImage(&rndrTexRes, param.texID, GL_TEXTURE_2D,
                                          cudaGraphicsRegisterFlagsWriteDiscard));
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
}

void setDPBXParam(const dpbxvdb::VDBInfo &vdbInfo_, const dpbxvdb::VDBDeviceData &vdbDat) {
    vdbInfo = vdbInfo_;
    CUDACheck(cudaMemcpyToSymbol(dc_vdbInfo, &vdbInfo, sizeof(vdbInfo)));
    CUDACheck(cudaMemcpyToSymbol(dc_vdbDat, &vdbDat, sizeof(vdbDat)));
}

void setTF(const std::vector<float> &flatTF) {
    if (tf != 0)
        CUDACheck(cudaDestroyTextureObject(tf));

    auto chnlDesc = cudaCreateChannelDesc<float4>();
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    CUDACheck(cudaMallocArray(&resDesc.res.array.array, &chnlDesc, flatTF.size() >> 2));
    CUDACheck(cudaMemcpyToArray(resDesc.res.array.array, 0, 0, flatTF.data(),
                                sizeof(float) * flatTF.size(), cudaMemcpyHostToDevice));

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CUDACheck(cudaCreateTextureObject(&tf, &resDesc, &texDesc, nullptr));
    CUDACheck(cudaMemcpyToSymbol(dc_tf, &tf, sizeof(tf)));
}

inline __device__ uchar4 glmVec3ToUChar4(glm::vec3 v3) {
    v3.x = __saturatef(v3.x); // clamp to [0, 1]
    v3.y = __saturatef(v3.y);
    v3.z = __saturatef(v3.z);
    v3.x *= 255.f;
    v3.y *= 255.f;
    v3.z *= 255.f;
    return make_uchar4(v3.x, v3.y, v3.z, 255);
}

inline __device__ glm::vec4 transferFunc(float v) {
    if (v < dc_rndr.thresh)
        return glm::zero<glm::vec4>();
    auto rgba = tex1D<float4>(dc_tf, v);
    return glm::vec4{rgba.x, rgba.y, rgba.z, rgba.w};
}

inline __device__ glm::vec3 pix2RayDir(const glm::uvec2 &pix, const glm::mat4 &unProj,
                                       const glm::mat4 &tr) {
    glm::vec4 tmp{float((pix.x << 1) + 1) / dc_rndr.res.x - 1.f,
                  float((pix.y << 1) + 1) / dc_rndr.res.y - 1.f, 1.f, 1.f};
    tmp = unProj * tmp;
    glm::vec3 rayDir = tmp;

    rayDir = glm::mat3{tr} * glm::normalize(rayDir);
    return rayDir;
}

inline __device__ void integralColor(glm::vec3 &rgb, float &a, const glm::vec3 &LR,
                                     const glm::vec3 &posInBrick, const glm::vec3 &aMin) {
    if (dc_rndr.usePhongShading) {
        auto v = tex3D<float>(dc_vdbDat.atlasTex, aMin.x + posInBrick.x, aMin.y + posInBrick.y,
                              aMin.z + posInBrick.z);
        auto N = [&]() {
            glm::vec3 ret;
            float v0, v1;
#pragma unroll
            for (uint8_t xyz = 0; xyz < 3; ++xyz) {
                v0 =
                    tex3D<float>(dc_vdbDat.atlasTex, aMin.x + posInBrick.x + (xyz == 0 ? 1.f : 0.f),
                                 aMin.y + posInBrick.y + (xyz == 1 ? 1.f : 0.f),
                                 aMin.z + posInBrick.z + (xyz == 2 ? 1.f : 0.f));
                v1 =
                    tex3D<float>(dc_vdbDat.atlasTex, aMin.x + posInBrick.x - (xyz == 0 ? 1.f : 0.f),
                                 aMin.y + posInBrick.y - (xyz == 1 ? 1.f : 0.f),
                                 aMin.z + posInBrick.z - (xyz == 2 ? 1.f : 0.f));
                ret[xyz] = v1 - v0;
            }
            return ret;
        }();
        if (glm::dot(N, LR) < 0.f)
            N *= -1.f;

        auto col = transferFunc(v);
        auto diffCol = glm::vec3{col};
        auto ambient = dc_rndr.ka * diffCol;
        auto NdtLR = glm::dot(N, LR);
        auto specular = dc_rndr.ks * glm::pow(NdtLR, dc_rndr.shininess);
        auto diffuse = dc_rndr.kd * NdtLR * diffCol;
#pragma unroll
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            diffCol[xyz] = col[xyz] = ambient[xyz] + diffuse[xyz] + specular;

        rgb = rgb + (1.f - a) * col.a * diffCol;
        a = a + (1.f - a) * col.a;
    } else {
        auto v = tex3D<float>(dc_vdbDat.atlasTex, aMin.x + posInBrick.x, aMin.y + posInBrick.y,
                              aMin.z + posInBrick.z);
        auto col = transferFunc(v);
        auto diffCol = glm::vec3{col};

        rgb = rgb + (1.f - a) * col.a * diffCol;
        a = a + (1.f - a) * col.a;
    }
}

inline __device__ void mixForeBackGround(glm::vec3 &rgb, float a) {
    rgb = a * rgb + (1.f - a) * dc_rndr.bkgrndCol;
}

inline __device__ bool trySkipBirck(float &t, const glm::vec3 &rayPos, const glm::vec3 &rayDir,
                                    float tMax, const glm::vec3 &posInBrick,
                                    const glm::vec3 &aMin) {
    using namespace dpbxvdb;

    DPBXDDA2D dpbxdda;
    if (!dpbxdda.Init(rayPos, rayDir, t, posInBrick, dc_vdbInfo))
        return false;

    auto lastDep = tex3D<float>(dc_vdbDat.atlasDepTex, aMin.x + dpbxdda.idx3InBlock.x,
                                aMin.y + dpbxdda.idx3InBlock.y, aMin.z + dpbxdda.idx3InBlock.z);
    while (true) {
        dpbxdda.StepNext();
        auto dep = tex3D<float>(dc_vdbDat.atlasDepTex, aMin.x + dpbxdda.idx3InBlock.x,
                                aMin.y + dpbxdda.idx3InBlock.y, aMin.z + dpbxdda.idx3InBlock.z);
        if (glm::min(lastDep, dep) <= dpbxdda.dep)
            return false;
        if (dpbxdda.t > tMax)
            return true;

        t = glm::max(t, dpbxdda.t);
        lastDep = dep;
    }
    return false;
}

__global__ void renderKernel(cudaSurfaceObject_t outSurf, glm::mat4 unProj, glm::mat4 tr) {
    using namespace dpbxvdb;

    glm::uvec2 pix{blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
    if (pix.x >= dc_rndr.res.x || pix.y >= dc_rndr.res.y)
        return;

    glm::vec3 camPos = tr[3];
    auto rayDir = pix2RayDir(pix, unProj, tr);
    auto LR = -rayDir;
    const auto AABBMax = glm::vec3(dc_vdbInfo.voxPerVol);
    auto tStart = rayIntersectAABB(camPos, rayDir, glm::zero<glm::vec3>(), AABBMax);
    if (tStart.z < 0.f) {
        surf2Dwrite(glmVec3ToUChar4(dc_rndr.bkgrndCol), outSurf, pix.x * sizeof(uchar4), pix.y);
        return;
    }

    auto lev = dc_vdbInfo.topLev;
    float voxPerBrick = dc_vdbInfo.dims[0];
    float tMaxStk[MaxLevNum];
    tStart.x += Epsilon;
    tMaxStk[lev] = tStart.y - Epsilon;

    IDTy nodeIdxStk[MaxLevNum];
    auto &node = dc_vdbDat.nodePools[lev][0];
    nodeIdxStk[lev] = 0;

    HDDA3D hdda;
    hdda.Init(camPos, rayDir, tStart);
    hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);

    auto rgb = glm::zero<glm::vec3>();
    auto alpha = 0.f;

    auto brickFn = [&]() {
        auto t = dc_rndr.dt * glm::ceil(hdda.t.x / dc_rndr.dt);
        auto &leaf = dc_vdbDat.nodePools[0][nodeIdxStk[0]];
        glm::vec3 aMin = leaf.atlasIdx3;
        auto posInBrick = camPos + t * rayDir - glm::vec3{leaf.idx3};
        auto dPos = dc_rndr.dt * rayDir;

        while (true) {
            auto stop = t > tMaxStk[0];
#pragma unroll
            for (uint8_t xyz = 0; xyz < 3; ++xyz)
                if (posInBrick[xyz] < 0.f || posInBrick[xyz] >= voxPerBrick) {
                    stop = true;
                    break;
                }
            if (stop)
                break;

            integralColor(rgb, alpha, LR, posInBrick, aMin);

            t += dc_rndr.dt;
            posInBrick += dPos;
        }
    };

    while (true) {
        auto stop = lev > dc_vdbInfo.topLev;
#pragma unroll
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (hdda.chIdx3[xyz] < 0 || hdda.chIdx3[xyz] >= dc_vdbInfo.dims[lev]) {
                stop = true;
                break;
            }
        if (stop)
            break;

        hdda.Next();
        auto chIdx = dc_vdbInfo.PosInBrick2ChildIdx(hdda.chIdx3, lev);
        if (IsChildOn(dc_vdbDat, node, chIdx)) {
            if (lev == 1) {
                nodeIdxStk[0] = Node::ID2Idx(GetChildID(dc_vdbDat, node, chIdx));
                hdda.t.x += Epsilon;
                tMaxStk[0] = hdda.t.y - Epsilon;
                brickFn();
                hdda.Step();
            } else {
                --lev;
                nodeIdxStk[lev] = Node::ID2Idx(GetChildID(dc_vdbDat, node, chIdx));
                node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                hdda.t.x += Epsilon;
                tMaxStk[lev] = hdda.t.y - Epsilon;
                hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
            }
        } else
            hdda.Step();

        while (hdda.t.x > tMaxStk[lev] && lev <= dc_vdbInfo.topLev) {
            ++lev;
            if (lev <= dc_vdbInfo.topLev) {
                node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
            }
        }
    }

#pragma unroll
    for (uint8_t xyz = 0; xyz < 3; ++xyz)
        rgb[xyz] = glm::pow(rgb[xyz], GammaCorrCoef);
    mixForeBackGround(rgb, alpha);
    surf2Dwrite(glmVec3ToUChar4(rgb), outSurf, pix.x * sizeof(uchar4), pix.y);
}

__global__ void renderDPBXKernel(cudaSurfaceObject_t outSurf, glm::mat4 unProj, glm::mat4 tr) {
    using namespace dpbxvdb;

    glm::uvec2 pix{blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
    if (pix.x >= dc_rndr.res.x || pix.y >= dc_rndr.res.y)
        return;

    glm::vec3 camPos = tr[3];
    auto rayDir = pix2RayDir(pix, unProj, tr);
    auto LR = -rayDir;
    const auto AABBMax = glm::vec3(dc_vdbInfo.voxPerVol);
    auto tStart = rayIntersectAABB(camPos, rayDir, glm::zero<glm::vec3>(), AABBMax);
    if (tStart.z < 0.f) {
        surf2Dwrite(glmVec3ToUChar4(dc_rndr.bkgrndCol), outSurf, pix.x * sizeof(uchar4), pix.y);
        return;
    }

    auto lev = dc_vdbInfo.topLev;
    float voxPerBrick = dc_vdbInfo.dims[0];
    float tMaxStk[MaxLevNum];
    tStart.x += Epsilon;
    tMaxStk[lev] = tStart.y - Epsilon;

    IDTy nodeIdxStk[MaxLevNum];
    auto &node = dc_vdbDat.nodePools[lev][0];
    nodeIdxStk[lev] = 0;

    HDDA3D hdda;
    hdda.Init(camPos, rayDir, tStart);
    hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);

    auto rgb = glm::zero<glm::vec3>();
    auto alpha = 0.f;

    auto brickFn = [&]() {
        auto t = dc_rndr.dt * glm::ceil(hdda.t.x / dc_rndr.dt);
        auto &leaf = dc_vdbDat.nodePools[0][nodeIdxStk[0]];
        glm::vec3 aMin = leaf.atlasIdx3;
        glm::vec3 vMin = leaf.idx3;
        auto posInBrick = camPos + t * rayDir - vMin;
        auto dPos = dc_rndr.dt * rayDir;

        auto skipBrick = trySkipBirck(t, camPos, rayDir, tMaxStk[0], posInBrick, aMin);
        if (skipBrick)
            return;

        posInBrick = camPos + t * rayDir - vMin;
        while (true) {
            auto stop = t > tMaxStk[0];
#pragma unroll
            for (uint8_t xyz = 0; xyz < 3; ++xyz)
                if (posInBrick[xyz] < 0.f || posInBrick[xyz] >= voxPerBrick) {
                    stop = true;
                    break;
                }
            if (stop)
                break;

            integralColor(rgb, alpha, LR, posInBrick, aMin);

            t += dc_rndr.dt;
            posInBrick += dPos;
        }
    };

    while (true) {
        auto stop = lev > dc_vdbInfo.topLev;
#pragma unroll
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (hdda.chIdx3[xyz] < 0 || hdda.chIdx3[xyz] >= dc_vdbInfo.dims[lev]) {
                stop = true;
                break;
            }
        if (stop)
            break;

        hdda.Next();
        auto chIdx = dc_vdbInfo.PosInBrick2ChildIdx(hdda.chIdx3, lev);
        if (IsChildOn(dc_vdbDat, node, chIdx)) {
            if (lev == 1) {
                nodeIdxStk[0] = Node::ID2Idx(GetChildID(dc_vdbDat, node, chIdx));
                hdda.t.x += Epsilon;
                tMaxStk[0] = hdda.t.y - Epsilon;
                brickFn();
                hdda.Step();
            } else {
                --lev;
                nodeIdxStk[lev] = Node::ID2Idx(GetChildID(dc_vdbDat, node, chIdx));
                node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                hdda.t.x += Epsilon;
                tMaxStk[lev] = hdda.t.y - Epsilon;
                hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
            }
        } else
            hdda.Step();

        while (hdda.t.x > tMaxStk[lev] && lev <= dc_vdbInfo.topLev) {
            ++lev;
            if (lev <= dc_vdbInfo.topLev) {
                node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
            }
        }
    }

#pragma unroll
    for (uint8_t xyz = 0; xyz < 3; ++xyz)
        rgb[xyz] = glm::pow(rgb[xyz], GammaCorrCoef);
    mixForeBackGround(rgb, alpha);
    surf2Dwrite(glmVec3ToUChar4(rgb), outSurf, pix.x * sizeof(uchar4), pix.y);
}

__global__ void renderDepthKernel(cudaSurfaceObject_t outSurf, glm::mat4 unProj, glm::mat4 tr) {
    using namespace dpbxvdb;

    glm::uvec2 pix{blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
    if (pix.x >= dc_rndr.res.x || pix.y >= dc_rndr.res.y)
        return;

    glm::vec3 camPos = tr[3];
    auto rayDir = pix2RayDir(pix, unProj, tr);
    const auto AABBMax = glm::vec3(dc_vdbInfo.voxPerVol);
    auto tStart = rayIntersectAABB(camPos, rayDir, glm::zero<glm::vec3>(), AABBMax);
    if (!dc_vdbInfo.useDPBX || tStart.z < 0.f) {
        surf2Dwrite(glmVec3ToUChar4(dc_rndr.bkgrndCol), outSurf, pix.x * sizeof(uchar4), pix.y);
        return;
    }

    auto lev = dc_vdbInfo.topLev;
    float voxPerBrick = dc_vdbInfo.dims[0];
    float tMaxStk[MaxLevNum];
    tStart.x += Epsilon;
    tMaxStk[lev] = tStart.y - Epsilon;

    IDTy nodeIdxStk[MaxLevNum];
    auto &node = dc_vdbDat.nodePools[lev][0];
    nodeIdxStk[lev] = 0;

    HDDA3D hdda;
    hdda.Init(camPos, rayDir, tStart);
    hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);

    auto rgb = glm::zero<glm::vec3>();
    auto alpha = 0.f;
    while (true) {
        auto stop = lev > dc_vdbInfo.topLev;
#pragma unroll
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (hdda.chIdx3[xyz] < 0 || hdda.chIdx3[xyz] >= dc_vdbInfo.dims[lev]) {
                stop = true;
                break;
            }
        if (stop)
            break;

        hdda.Next();
        auto chIdx = dc_vdbInfo.PosInBrick2ChildIdx(hdda.chIdx3, lev);
        if (IsChildOn(dc_vdbDat, node, chIdx)) {
            if (lev == 1) {
                nodeIdxStk[0] = Node::ID2Idx(GetChildID(dc_vdbDat, node, chIdx));
                hdda.t.x += Epsilon;

                auto &leaf = dc_vdbDat.nodePools[0][nodeIdxStk[0]];
                glm::vec3 aMin = leaf.atlasIdx3;
                auto posInBrick = camPos + hdda.t.x * rayDir - glm::vec3{leaf.idx3};

                DPBXDDA2D dpbxdda;
                dpbxdda.Init(camPos, rayDir, hdda.t.x, posInBrick, dc_vdbInfo);
                auto dep =
                    tex3D<float>(dc_vdbDat.atlasDepTex, aMin.x + dpbxdda.idx3InBlock.x,
                                 aMin.y + dpbxdda.idx3InBlock.y, aMin.z + dpbxdda.idx3InBlock.z);

                rgb = glm::vec3{dep / voxPerBrick};
                alpha = 1.f;
                break;
            } else {
                --lev;
                nodeIdxStk[lev] = Node::ID2Idx(GetChildID(dc_vdbDat, node, chIdx));
                node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                hdda.t.x += Epsilon;
                tMaxStk[lev] = hdda.t.y - Epsilon;
                hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
            }
        } else
            hdda.Step();

        while (hdda.t.x > tMaxStk[lev] && lev <= dc_vdbInfo.topLev) {
            ++lev;
            if (lev <= dc_vdbInfo.topLev) {
                node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
            }
        }
    }

    mixForeBackGround(rgb, alpha);
    surf2Dwrite(glmVec3ToUChar4(rgb), outSurf, pix.x * sizeof(uchar4), pix.y);
}

__global__ void renderSkipTimeDiffKernel(cudaSurfaceObject_t outSurf, glm::mat4 unProj,
                                         glm::mat4 tr) {
    using namespace dpbxvdb;

    glm::uvec2 pix{blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
    if (pix.x >= dc_rndr.res.x || pix.y >= dc_rndr.res.y)
        return;

    glm::vec3 camPos = tr[3];
    auto rayDir = pix2RayDir(pix, unProj, tr);
    const auto AABBMax = glm::vec3(dc_vdbInfo.voxPerVol);
    auto tStart = rayIntersectAABB(camPos, rayDir, glm::zero<glm::vec3>(), AABBMax);
    if (!dc_vdbInfo.useDPBX || tStart.z < 0.f) {
        surf2Dwrite(glmVec3ToUChar4(dc_rndr.bkgrndCol), outSurf, pix.x * sizeof(uchar4), pix.y);
        return;
    }

    auto lev = dc_vdbInfo.topLev;
    float voxPerBrick = dc_vdbInfo.dims[0];
    float tMaxStk[MaxLevNum];
    tStart.x += Epsilon;
    tMaxStk[lev] = tStart.y - Epsilon;

    IDTy nodeIdxStk[MaxLevNum];
    auto &node = dc_vdbDat.nodePools[lev][0];
    nodeIdxStk[lev] = 0;

    HDDA3D hdda;
    hdda.Init(camPos, rayDir, tStart);
    hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);

    auto rgb = glm::zero<glm::vec3>();
    auto alpha = 0.f;

    auto brickFn = [&]() {
        auto t = dc_rndr.dt * glm::ceil(hdda.t.x / dc_rndr.dt);
        auto &leaf = dc_vdbDat.nodePools[0][nodeIdxStk[0]];
        glm::vec3 aMin = leaf.atlasIdx3;
        glm::vec3 vMin = leaf.idx3;
        auto posInBrick = camPos + t * rayDir - vMin;
        auto dPos = dc_rndr.dt * rayDir;

        // find time given by DPBX method
        auto skipBrick = trySkipBirck(t, camPos, rayDir, tMaxStk[0], posInBrick, aMin);
        if (skipBrick)
            return;
        auto tDPBX = t;

        // find the real time to the first voxel larger than threshold
        t = dc_rndr.dt * glm::ceil(hdda.t.x / dc_rndr.dt);
        posInBrick = camPos + t * rayDir - vMin;
        while (true) {
            auto stop = t > tMaxStk[0];
#pragma unroll
            for (uint8_t xyz = 0; xyz < 3; ++xyz)
                if (posInBrick[xyz] < 0.f || posInBrick[xyz] >= voxPerBrick) {
                    stop = true;
                    break;
                }
            if (stop)
                break;

            auto v = tex3D<float>(dc_vdbDat.atlasTex, aMin.x + posInBrick.x, aMin.y + posInBrick.y,
                                  aMin.z + posInBrick.z);
            if (v >= dc_vdbInfo.thresh)
                break;

            t += dc_rndr.dt;
            posInBrick += dPos;
        }

        // show difference
        rgb = glm::zero<glm::vec3>();
        if (t >= tDPBX)
            rgb.r = rgb.g = rgb.b = (t - tDPBX) / voxPerBrick;
        else
            rgb.r = (tDPBX - t) / voxPerBrick;
        alpha = 1.f;
    };

    while (true) {
        auto stop = lev > dc_vdbInfo.topLev;
#pragma unroll
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            if (hdda.chIdx3[xyz] < 0 || hdda.chIdx3[xyz] >= dc_vdbInfo.dims[lev]) {
                stop = true;
                break;
            }
        if (stop)
            break;

        hdda.Next();
        auto chIdx = dc_vdbInfo.PosInBrick2ChildIdx(hdda.chIdx3, lev);
        if (IsChildOn(dc_vdbDat, node, chIdx)) {
            if (lev == 1) {
                nodeIdxStk[0] = Node::ID2Idx(GetChildID(dc_vdbDat, node, chIdx));
                hdda.t.x += Epsilon;
                tMaxStk[0] = hdda.t.y - Epsilon;
                brickFn();
                if (alpha != 0.f)
                    break;
                hdda.Step();
            } else {
                --lev;
                nodeIdxStk[lev] = Node::ID2Idx(GetChildID(dc_vdbDat, node, chIdx));
                node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                hdda.t.x += Epsilon;
                tMaxStk[lev] = hdda.t.y - Epsilon;
                hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
            }
        } else
            hdda.Step();

        while (hdda.t.x > tMaxStk[lev] && lev <= dc_vdbInfo.topLev) {
            ++lev;
            if (lev <= dc_vdbInfo.topLev) {
                node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
            }
        }
    }

    mixForeBackGround(rgb, alpha);
    surf2Dwrite(glmVec3ToUChar4(rgb), outSurf, pix.x * sizeof(uchar4), pix.y);
}

__global__ void renderBrickKernel(cudaSurfaceObject_t outSurf, glm::mat4 unProj, glm::mat4 tr,
                                  uint8_t drawLev) {
    using namespace dpbxvdb;

    glm::uvec2 pix{blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y};
    if (pix.x >= dc_rndr.res.x || pix.y >= dc_rndr.res.y)
        return;

    glm::vec3 camPos = tr[3];
    auto rayDir = pix2RayDir(pix, unProj, tr);
    const auto AABBMax = glm::vec3(dc_vdbInfo.voxPerVol);
    auto tStart = rayIntersectAABB(camPos, rayDir, glm::zero<glm::vec3>(), AABBMax);
    if (tStart.z < 0.f) {
        surf2Dwrite(glmVec3ToUChar4(dc_rndr.bkgrndCol), outSurf, pix.x * sizeof(uchar4), pix.y);
        return;
    }

    auto lev = dc_vdbInfo.topLev;
    float tMaxStk[MaxLevNum];
    tStart.x += Epsilon;
    tMaxStk[lev] = tStart.y - Epsilon;

    IDTy nodeIdxStk[MaxLevNum];
    auto &node = dc_vdbDat.nodePools[lev][0];
    nodeIdxStk[lev] = 0;

    HDDA3D hdda;
    hdda.Init(camPos, rayDir, tStart);
    hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);

    auto rgb = glm::zero<glm::vec3>();
    auto alpha = 0.f;
    if (drawLev == lev) {
        auto p = camPos + hdda.t.x * rayDir;
        glm::vec3 rng = dc_vdbInfo.voxPerVol;
        rgb = p / rng;
        alpha = 1.f;
    } else
        while (true) {
            auto stop = lev > dc_vdbInfo.topLev;
#pragma unroll
            for (uint8_t xyz = 0; xyz < 3; ++xyz)
                if (hdda.chIdx3[xyz] < 0 || hdda.chIdx3[xyz] >= dc_vdbInfo.dims[lev]) {
                    stop = true;
                    break;
                }
            if (stop)
                break;

            hdda.Next();
            auto chIdx = dc_vdbInfo.PosInBrick2ChildIdx(hdda.chIdx3, lev);
            if (IsChildOn(dc_vdbDat, node, chIdx)) {
                if (lev == drawLev + 1 || lev <= 1) {
                    auto p = camPos + hdda.t.x * rayDir;
                    glm::vec3 rng = dc_vdbInfo.voxPerVol;
                    rgb = p / rng;
                    alpha = 1.f;
                    break;
                } else {
                    --lev;
                    nodeIdxStk[lev] = Node::ID2Idx(GetChildID(dc_vdbDat, node, chIdx));
                    node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                    hdda.t.x += Epsilon;
                    tMaxStk[lev] = hdda.t.y - Epsilon;
                    hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
                }
            } else
                hdda.Step();

            while (hdda.t.x > tMaxStk[lev] && lev <= dc_vdbInfo.topLev) {
                ++lev;
                if (lev <= dc_vdbInfo.topLev) {
                    node = dc_vdbDat.nodePools[lev][nodeIdxStk[lev]];
                    hdda.Prepare(node.idx3, dc_vdbInfo.vDlts[lev]);
                }
            }
        }

    mixForeBackGround(rgb, alpha);
    surf2Dwrite(glmVec3ToUChar4(rgb), outSurf, pix.x * sizeof(uchar4), pix.y);
}

void render(const glm::mat4 &unProj, const glm::mat4 &tr, RenderTarget rndrTarget,
            float &costInMs) {
    cudaGraphicsMapResources(1, &rndrTexRes);

    cudaGraphicsSubResourceGetMappedArray(&resDesc.res.array.array, rndrTexRes, 0, 0);
    cudaSurfaceObject_t outSurf;
    cudaCreateSurfaceObject(&outSurf, &resDesc);

    dim3 block{16, 16};
    dim3 grid{((decltype(dim3::x))rndr.res.x + block.x - 1) / block.x,
              ((decltype(dim3::y))rndr.res.y + block.y - 1) / block.y};

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    switch (rndrTarget) {
    case RenderTarget::Vol:
        if (vdbInfo.useDPBX)
            renderDPBXKernel<<<grid, block>>>(outSurf, unProj, tr);
        else
            renderKernel<<<grid, block>>>(outSurf, unProj, tr);
        break;
    case RenderTarget::Depth:
        renderDepthKernel<<<grid, block>>>(outSurf, unProj, tr);
        break;
    case RenderTarget::SkipTimeDiff:
        renderSkipTimeDiffKernel<<<grid, block>>>(outSurf, unProj, tr);
        break;
    case RenderTarget::BrickL0:
    case RenderTarget::BrickL1:
    case RenderTarget::BrickL2:
        renderBrickKernel<<<grid, block>>>(outSurf, unProj, tr,
                                           static_cast<uint8_t>(rndrTarget) -
                                               static_cast<uint8_t>(RenderTarget::BrickL0));
        break;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    costInMs = 0.f;
    cudaEventElapsedTime(&costInMs, start, stop);

    cudaDestroySurfaceObject(outSurf);
    cudaGraphicsUnmapResources(1, &rndrTexRes);
}
