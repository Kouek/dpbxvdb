#include <chrono>
#include <format>
#include <iostream>
#include <string_view>

#include <dpbxvdb/dpbxvdb.h>

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <cg/fps_camera.h>
#include <cg/imgui_tf.h>
#include <cg/math.h>
#include <cmdparser.hpp>
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
static RenderTarget rndrTarget = RenderTarget::Vol;

static auto isSparse = true;
static auto useDPBX = true;
static std::array<uint8_t, 3> log2Dims{4, 5, 5};
static float costInMs = 0.f;

static auto reRndr = true;
constexpr auto ReRndrCntDownInterval = static_cast<std::chrono::milliseconds>(100);
static auto reRndrCntDownOn = false;
static auto reRndrCntDownStart = std::chrono::system_clock::time_point();

static kouek::FPSCamera camera;
static auto voxSpacing = glm::one<glm::vec3>();
static glm::vec3 invVoxSpacing;
static auto camFarClip = 10.f;
static auto cam2CntrDist = 0.f;
static glm::mat4 camUnProj;

inline void startReRndrCntDown() {
    reRndrCntDownOn = true;
    reRndrCntDownStart = std::chrono::system_clock::now();
}

static void framebufferSizeCallback(GLFWwindow *window, int width, int height) {
    if (rndr.texID != 0)
        glDeleteTextures(1, &rndr.texID);
    rndr.res.x = width;
    rndr.res.y = height;
    rndr.bkgrndCol = glm::vec3{.3f, .3f, .3f};
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

    camUnProj = kouek::Math::InverseProjective(glm::perspectiveFov(
        glm::radians(60.f), (float)width, (float)height, .01f * camFarClip, camFarClip));

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

    if (action != GLFW_RELEASE)
        switch (key) {
        case GLFW_KEY_UP:
        case GLFW_KEY_DOWN:
        case GLFW_KEY_LEFT:
        case GLFW_KEY_RIGHT:
        case GLFW_KEY_Q:
        case GLFW_KEY_E:
            cam2CntrDist = glm::distance(camera.GetPos(), .5f * voxSpacing * glm::vec3{volDim});
            break;
        }

    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        break;
    case GLFW_KEY_UP:
        if (action != GLFW_RELEASE)
            camera.Revolve(cam2CntrDist, 0.f, +rotSens);
        break;
    case GLFW_KEY_DOWN:
        if (action != GLFW_RELEASE)
            camera.Revolve(cam2CntrDist, 0.f, -rotSens);
        break;
    case GLFW_KEY_LEFT:
        if (action != GLFW_RELEASE)
            camera.Revolve(cam2CntrDist, +rotSens, 0.f);
        break;
    case GLFW_KEY_RIGHT:
        if (action != GLFW_RELEASE)
            camera.Revolve(cam2CntrDist, -rotSens, 0.f);
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
    if (isSparse)
        volume.LoadAsSparse(volPath, volDim, volThresh, useDPBX, log2Dims);
    else
        volume.LoadAsDense(volPath, volDim, useDPBX, log2Dims);
    static auto first = true;
    if (first) {
        camera.LookAt(glm::vec3{.5f, .5f, 1.2f} * voxSpacing * glm::vec3{volDim},
                      .5f * voxSpacing * glm::vec3{volDim});
        camFarClip = 1.7f * std::max({voxSpacing.x * volDim.x, voxSpacing.y * volDim.y,
                                      voxSpacing.z * volDim.z});

        first = false;
    }

    setDPBXParam(volume.GetVDB().GetInfo(), volume.GetVDB().GetDeviceData());
}

int main(int argc, char **argv) {
    cli::Parser parser(argc, argv);
    parser.set_required<std::string>("vol", "volume", "Path of raw volume file");
    parser.set_required<glm::uint>("dx", "dim-x", "Dimension of volume on X-axis");
    parser.set_required<glm::uint>("dy", "dim-y", "Dimension of volume on Y-axis");
    parser.set_required<glm::uint>("dz", "dim-z", "Dimension of volume on Z-axis");
    parser.set_required<DpbxRawVoxTy>("th", "threshold",
                                      "Threshold of voxel density to denoise volume");
    parser.set_optional<std::string>("tf", "transfer-func", "",
                                     "Path of TXT file storing transfer function");
    parser.set_optional<float>("sx", "spacing-x", -1.f, "Spacing of voxel on X-axis");
    parser.set_optional<float>("sy", "spacing-y", -1.f, "Spacing of voxel on Y-axis");
    parser.set_optional<float>("sz", "spacing-z", -1.f, "Spacing of voxel on X-axis");
    parser.run_and_exit_if_error();

    volPath = parser.get<std::string>("vol");
    volDim.x = parser.get<glm::uint>("dx");
    volDim.y = parser.get<glm::uint>("dy");
    volDim.z = parser.get<glm::uint>("dz");
    volThresh = parser.get<DpbxRawVoxTy>("th");

    if (auto sx = parser.get<float>("sx"); sx != -1.f)
        voxSpacing.x = sx;
    if (auto sy = parser.get<float>("sy"); sy != -1.f)
        voxSpacing.y = sy;
    if (auto sz = parser.get<float>("sz"); sz != -1.f)
        voxSpacing.z = sz;
    invVoxSpacing = 1.f / voxSpacing;

    rndr.usePhongShading = false;
    rndr.ka = rndr.kd = rndr.ks = .5f;
    rndr.shininess = 64.f;

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

    if (auto tfPath = parser.get<std::string>("tf"); !tfPath.empty())
        try {
            tfWidget.Load(tfPath);
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
        framebufferSizeCallback(window, w, h);
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
            auto tr = glm::scale(glm::identity<glm::mat4>(), invVoxSpacing);
            tr = tr * glm::mat4{R.x,  R.y,  R.z,  0.f, U.x, U.y, U.z, 0.f,
                                -F.x, -F.y, -F.z, 0.f, P.x, P.y, P.z, 1.f};
            render(camUnProj, tr, rndrTarget, costInMs);
            reRndr = false;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Efficiency and Camera Setting");
        {
            ImGui::Text("Cost %f ms, with %f fps", costInMs, 1000.f / costInMs);
            if (ImGui::DragFloat3("Voxel Spacing", &voxSpacing[0], .1f, dpbxvdb::Epsilon,
                                  std::numeric_limits<float>::infinity())) {
                invVoxSpacing = 1.f / voxSpacing;
                reRndr = true;
            }
        }
        ImGui::End();
        ImGui::Begin("Rendering Setting");
        {
            if (tfWidget()) {
                changeTF();
                startReRndrCntDown();
            }
            static auto lastSelected = static_cast<uint8_t>(rndrTarget);
            for (uint8_t i = 0; i < static_cast<uint8_t>(RenderTarget::End); ++i) {
                if (i != 0)
                    ImGui::SameLine();
                if (ImGui::RadioButton(RenderTargetNames[i], lastSelected == i))
                    if (lastSelected != i) {
                        rndrTarget = static_cast<RenderTarget>(i);
                        lastSelected = i;
                        reRndr = true;
                    }
            }
            {
                bool reSetRndrParam = false;
                if (ImGui::RadioButton("Use Phong Shading", rndr.usePhongShading)) {
                    rndr.usePhongShading = !rndr.usePhongShading;
                    reSetRndrParam = true;
                }
                ImGui::PushItemWidth(100.f);
                if (ImGui::InputFloat("ka", &rndr.ka, .05f, .1f))
                    reSetRndrParam = true;
                ImGui::SameLine();
                if (ImGui::InputFloat("kd", &rndr.kd, .05f, .1f))
                    reSetRndrParam = true;
                ImGui::SameLine();
                if (ImGui::InputFloat("ks", &rndr.ks, .05f, .1f))
                    reSetRndrParam = true;
                ImGui::SameLine();
                if (ImGui::InputFloat("shininess", &rndr.shininess, 0.f, 0.f))
                    reSetRndrParam = true;
                if (reSetRndrParam) {
                    int w, h;
                    glfwGetFramebufferSize(window, &w, &h);
                    framebufferSizeCallback(window, w, h);
                    reRndr = true;
                }
                ImGui::PopItemWidth();
            }
        }
        ImGui::End();
        ImGui::Begin("Tree Setting");
        {
            if (ImGui::RadioButton("Is Sparse", isSparse)) {
                isSparse = !isSparse;
                loadVol();
                reRndr = true;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Use DPBX", useDPBX)) {
                useDPBX = !useDPBX;
                loadVol();
                reRndr = true;
            }

            ImGui::Text("log2 Dims: <");
            ImGui::SameLine();
            ImGui::PushItemWidth(100.f);
            int leafLog2Dim = log2Dims[0];
            if (ImGui::InputInt("###", &leafLog2Dim, 1))
                if (leafLog2Dim >= 3 && leafLog2Dim <= 10) {
                    log2Dims[0] = leafLog2Dim;
                    loadVol();
                    reRndr = true;
                }
            ImGui::PopItemWidth();
            ImGui::SameLine();
            ImGui::Text("5,5>");
        }
        ImGui::End();
        ImGui::EndFrame();
        ImGui::Render();

        glViewport(0, 0, rndr.res.x, rndr.res.y);
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
