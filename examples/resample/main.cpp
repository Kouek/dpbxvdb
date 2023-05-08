#include <chrono>
#include <iostream>
#include <string_view>

#include <dpbxvdb/dpbxvdb.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <cg/fps_camera.h>
#include <cg/imgui_tf.h>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "gl_helper.h"
#include "raw_volume_loader.h"
#include "render.h"

static RawVolume<DpbxRawVoxTy> volume;
static std::string volPath;
static glm::uvec3 volDim;
static DpbxRawVoxTy volThresh;

static RenderParam rndr;
static RenderTarget rndrTarget = RenderTarget::SparseVol;

static auto reRndr = true;
constexpr auto ReRndrCntDownInterval = static_cast<std::chrono::milliseconds>(100);
static auto reRndrCntDownOn = false;
static auto reRndrCntDownStart = std::chrono::system_clock::time_point();

static kouek::FPSCamera camera;
static auto camFarClip = 10.f;
static auto camDist2Cntr = 0.f;

inline void startReRndrCntDown() {
    reRndrCntDownOn = true;
    reRndrCntDownStart = std::chrono::system_clock::now();
}

static void changeRenderParam(int width, int height) {
    auto sz = std::min(width, height);
    width = height = sz;

    if (rndr.texID != 0)
        glDeleteTextures(1, &rndr.texID);
    rndr.res.x = width;
    rndr.res.y = height;
    rndr.proj = glm::perspectiveFov(glm::radians(60.f), (float)width, (float)height,
                                    .01f * camFarClip, camFarClip);
    rndr.bkgrndCol = glm::vec3{.1f, .1f, .2f};
    rndr.thresh = volThresh / (float)std::numeric_limits<DpbxRawVoxTy>::max();
    rndr.dt = .25f;

    glGenTextures(1, &rndr.texID);
    glBindTexture(GL_TEXTURE_2D, rndr.texID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, rndr.res.x, rndr.res.y, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 (const void *)0);
    glBindTexture(GL_TEXTURE_2D, 0);

    setRenderParam(rndr);
}

static void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
    changeRenderParam(width, height);
    startReRndrCntDown();
}

static void windowCloseCallback(GLFWwindow *window) {
    release();

    if (rndr.texID != 0)
        glDeleteTextures(1, &rndr.texID);
}

static void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    const auto rotSens = glm::radians(30.f);
    const auto movSens = .02f * camFarClip;
    const auto &[R, F, U, P] = camera.GetRFUP();

    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;
    case GLFW_KEY_UP:
        if (action != GLFW_RELEASE)
            camera.Revolve(camDist2Cntr, 0.f, +rotSens);
        break;
    case GLFW_KEY_DOWN:
        if (action != GLFW_RELEASE)
            camera.Revolve(camDist2Cntr, 0.f, -rotSens);
        break;
    case GLFW_KEY_LEFT:
        if (action != GLFW_RELEASE)
            camera.Revolve(camDist2Cntr, +rotSens, 0.f);
        break;
    case GLFW_KEY_RIGHT:
        if (action != GLFW_RELEASE)
            camera.Revolve(camDist2Cntr, -rotSens, 0.f);
        break;
    case GLFW_KEY_Q:
        if (action != GLFW_RELEASE)
            camera.Move(0.f, 0.f, -movSens);
        break;
    case GLFW_KEY_E:
        if (action != GLFW_RELEASE)
            camera.Move(0.f, 0.f, +movSens);
        break;
    default:
        break;
    }

    if (action != GLFW_RELEASE)
        switch (key) {
        case GLFW_KEY_UP:
        case GLFW_KEY_DOWN:
        case GLFW_KEY_LEFT:
        case GLFW_KEY_RIGHT:
        case GLFW_KEY_Q:
        case GLFW_KEY_E:
            reRndr = true;
            break;
        }
}

static void loadVol() {
    switch (rndrTarget) {
    case RenderTarget::DenseVol:
        volume.LoadAsDense(volPath, volDim);
        break;
    case RenderTarget::SparseVol:
    case RenderTarget::TileL0:
    case RenderTarget::TileL1:
    case RenderTarget::TileL2:
        volume.LoadAsSparse(volPath, volDim, volThresh);
        break;
    }
    static auto first = true;
    if (first) {
        camera.LookAt({0.5f * volDim.x, 0.5f * volDim.y, 2.f * volDim.z},
                      {0.5f * volDim.x, 0.5f * volDim.y, 0.5f * volDim.z});
        camFarClip = 1.7f * std::max({volDim.x, volDim.y, volDim.z});
        camDist2Cntr = 1.5f * volDim.z;

        first = false;
    }
    
    setVolume();
}

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr
            << "Usage: " << argv[0]
            << " <raw_volume_path> <x_dim> <y_dim> <z_dim> <threshold> [transfer_fucntion_path]"
            << std::endl;
        return 1;
    }
    volPath = argv[1];
    volDim.x = std::atoi(argv[2]);
    volDim.y = std::atoi(argv[3]);
    volDim.z = std::atoi(argv[4]);
    volThresh = static_cast<DpbxRawVoxTy>(std::atoi(argv[5]));

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(900, 800, "Depth Box Resample", nullptr, nullptr);
    if (window == nullptr) {
        std::cerr << "Failed to glfwCreateWindow." << std::endl;
        return 1;
    }
    glfwMaximizeWindow(window);

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);
    glfwSetKeyCallback(window, keyCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD." << std::endl;
        glfwTerminate();
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    ImPlot::CreateContext();
    kouek::TFWidget<DpbxRawVoxTy> tfWidget;

    if (argc >= 7)
        try {
            tfWidget.Load(argv[6]);
        } catch (std::exception &e) {
            std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
        }
    auto changeTF = [&]() {
        auto &flatTF = tfWidget.GetFlatTF();
        setTF(flatTF);
    };
    changeTF();

    {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        changeRenderParam(w, h);
    }

    auto glRes = genGLRes();
    if (!glRes.shader->ok) {
        std::cerr << glRes.shader->errMsg << std::endl;
        goto TERMINAL;
    }

    try {
        loadVol();
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
        goto TERMINAL;
    }

    glUseProgram(glRes.shader->prog);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    while (!glfwWindowShouldClose(window)) {
        if (reRndrCntDownOn) {
            auto curr = std::chrono::system_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::milliseconds>(curr - reRndrCntDownStart);
            if (duration >= ReRndrCntDownInterval) {
                reRndr = true;
                reRndrCntDownOn = false;
            }
        }

        if (reRndr) {
            const auto &[R, F, U, P] = camera.GetRFUP();
            render(P, glm::mat3{R, U, -F}, rndrTarget, volume.GetVDB());
            reRndr = false;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            if (tfWidget()) {
                changeTF();
                startReRndrCntDown();
            }

            static auto lastSelected = static_cast<uint8_t>(rndrTarget);
            static auto lastIsDense = rndrTarget == RenderTarget::DenseVol;
            for (uint8_t i = 0; i < static_cast<uint8_t>(RenderTarget::End); ++i) {
                if (i != 0)
                    ImGui::SameLine();
                if (ImGui::RadioButton(RenderTargetNames[i], lastSelected == i))
                    if (lastSelected != i) {
                        rndrTarget = static_cast<RenderTarget>(i);
                        lastSelected = i;

                        if ((lastIsDense && rndrTarget != RenderTarget::DenseVol) ||
                            (!lastIsDense && rndrTarget == RenderTarget::DenseVol))
                            loadVol();
                        lastIsDense = rndrTarget == RenderTarget::DenseVol;

                        reRndr = true;
                    }
            }
        }
        ImGui::EndFrame();
        ImGui::Render();

        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, rndr.texID);
        glBindVertexArray(glRes.VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void *)0);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwPollEvents();
        glfwSwapBuffers(window);
    }

TERMINAL:
    ImPlot::DestroyContext();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    releaseGLRes(glRes);
    glfwTerminate();

    return 0;
}
