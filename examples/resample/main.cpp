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
static glm::uvec3 oldVoxPerVol;
static glm::uvec3 newVoxPerVol;
static DpbxRawVoxTy volThresh;
static dpbxvdb::AxisTransform volAxisTr{0, 1, 2};

static RenderParam rndr;
static RenderTarget rndrTarget = RenderTarget::Vol;

static auto drawUI = true;
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
    if (width == 0 || height == 0)
        return;

    if (rndr.texID != 0)
        glDeleteTextures(1, &rndr.texID);
    rndr.res.x = width;
    rndr.res.y = height;

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
            cam2CntrDist =
                glm::distance(camera.GetPos(), .5f * voxSpacing * glm::vec3{newVoxPerVol});
            break;
        case GLFW_KEY_U:
            drawUI = !drawUI;
            glfwSetInputMode(window, GLFW_CURSOR, drawUI ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_HIDDEN);
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
        volume.LoadAsSparse(volPath, oldVoxPerVol, volThresh, useDPBX, log2Dims, volAxisTr);
    else
        volume.LoadAsDense(volPath, oldVoxPerVol, useDPBX, log2Dims, volAxisTr);

    setDPBXParam(volume.GetVDB().GetInfo(), volume.GetVDB().GetDeviceData());
}

int main(int argc, char **argv) {
    cli::Parser parser(argc, argv);
    parser.set_required<std::string>("vol", "volume", "Path of raw volume file");
    parser.set_required<glm::uint>("dx", "dim-x",
                                   "Dimension of volume on X-axis (before axis transformation)");
    parser.set_required<glm::uint>("dy", "dim-y",
                                   "Dimension of volume on Y-axis (before axis transformation)");
    parser.set_required<glm::uint>("dz", "dim-z",
                                   "Dimension of volume on Z-axis (before axis transformation)");
    parser.set_required<DpbxRawVoxTy>("th", "threshold",
                                      "Threshold of voxel density to denoise volume");
    parser.set_optional<std::string>("tf", "transfer-func", "",
                                     "Path of TXT file storing transfer function");
    parser.set_optional<float>("sx", "spacing-x", -1.f, "Spacing of voxel on X-axis");
    parser.set_optional<float>("sy", "spacing-y", -1.f, "Spacing of voxel on Y-axis");
    parser.set_optional<float>("sz", "spacing-z", -1.f, "Spacing of voxel on Z-axis");
    {
        std::string trDesc(", 0 for X, 1 for Y and 2 for Z, while 3 for -X, 4 for -Y and 5 for -Z");
        parser.set_optional<uint8_t>("tx", "tr-x", 0, "Transform of X-axis" + trDesc);
        parser.set_optional<uint8_t>("ty", "tr-y", 1, "Transform of Y-axis" + trDesc);
        parser.set_optional<uint8_t>("tz", "tr-z", 2, "Transform of Z-axis" + trDesc);
        parser.run_and_exit_if_error();
    }

    volPath = parser.get<std::string>("vol");
    oldVoxPerVol.x = parser.get<glm::uint>("dx");
    oldVoxPerVol.y = parser.get<glm::uint>("dy");
    oldVoxPerVol.z = parser.get<glm::uint>("dz");
    volThresh = parser.get<DpbxRawVoxTy>("th");

    if (auto sx = parser.get<float>("sx"); sx != -1.f)
        voxSpacing.x = sx;
    if (auto sy = parser.get<float>("sy"); sy != -1.f)
        voxSpacing.y = sy;
    if (auto sz = parser.get<float>("sz"); sz != -1.f)
        voxSpacing.z = sz;
    invVoxSpacing = 1.f / voxSpacing;

    volAxisTr.x = parser.get<uint8_t>("tx");
    volAxisTr.y = parser.get<uint8_t>("ty");
    volAxisTr.z = parser.get<uint8_t>("tz");
    volAxisTr = [&]() {
        std::array<uint8_t, 3> hasVal{0};
        for (uint8_t xyz = 0; xyz < 3; ++xyz)
            ++hasVal[volAxisTr[xyz] % 3];
        for (auto v : hasVal)
            if (v == 0 || v >= 2) {
                std::cout << "Input option (tx, ty, tz) is invalid, (tx, ty, tz) is set to default "
                             "(0, 1, 2)"
                          << std::endl;
                return dpbxvdb::AxisTransform{0, 1, 2};
            }
        return volAxisTr;
    }();

    newVoxPerVol = volAxisTr.TransformDimension(oldVoxPerVol);

    rndr.dt = .25f;
    rndr.bkgrndCol = glm::vec3{.1f, .1f, .1f};
    rndr.usePhongShading = false;
    rndr.lightCol = glm::vec3{1.f, 1.f, 1.f};
    rndr.lightPos = voxSpacing * glm::vec3{.5f, 1.5f, .5f} * glm::vec3{newVoxPerVol};
    rndr.ka = .5f;
    rndr.kd = .5f;
    rndr.ks = .5f;
    rndr.shininess = 16.f;

    camera.LookAt(glm::vec3{.5f, .5f, 1.5f} * voxSpacing * glm::vec3{newVoxPerVol},
                  .5f * voxSpacing * glm::vec3{newVoxPerVol});
    camFarClip = 1.7f * std::max({voxSpacing.x * newVoxPerVol.x, voxSpacing.y * newVoxPerVol.y,
                                  voxSpacing.z * newVoxPerVol.z});

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
        auto reSetRndrParam = false;

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
        if (drawUI) {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::Begin("Efficiency and Camera Setting");
            {
                ImGui::Text("Cost %f ms, with %f fps", costInMs, 1000.f / costInMs);
                if (ImGui::DragFloat3("Voxel Spacing", &voxSpacing[0], .1f, dpbxvdb::Epsilon,
                                      std::numeric_limits<float>::infinity())) {
                    invVoxSpacing = 1.f / voxSpacing;
                    reSetRndrParam = true;
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
                    if (ImGui::RadioButton("Use Phong Shading", rndr.usePhongShading)) {
                        rndr.usePhongShading = !rndr.usePhongShading;
                        reSetRndrParam = true;
                    }

                    if (ImGui::InputFloat("ka", &rndr.ka, .05f, .1f))
                        reSetRndrParam = true;
                    if (ImGui::InputFloat("kd", &rndr.kd, .05f, .1f))
                        reSetRndrParam = true;
                    if (ImGui::InputFloat("ks", &rndr.ks, .05f, .1f))
                        reSetRndrParam = true;
                    if (ImGui::InputFloat("shininess", &rndr.shininess, 0.f, 0.f))
                        reSetRndrParam = true;
                    if (ImGui::ColorEdit3("light color", reinterpret_cast<float *>(&rndr.lightCol)))
                        reSetRndrParam = true;

                    ImGui::Text("light position:");
                    ImGui::PushItemWidth(100.f);
                    if (ImGui::InputFloat("x", &rndr.lightPos.x, 10.f, 50.f))
                        reSetRndrParam = true;
                    ImGui::SameLine();
                    if (ImGui::InputFloat("z", &rndr.lightPos.z, 10.f, 50.f))
                        reSetRndrParam = true;
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
        }

        glViewport(0, 0, rndr.res.x, rndr.res.y);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, rndr.texID);
        glBindVertexArray(glRes.VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void *)0);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);

        if (drawUI)
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (reSetRndrParam) {
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            framebufferSizeCallback(window, w, h);
            reRndr = true;
        }
        glfwPollEvents();
        glfwSwapBuffers(window);
    }

#ifdef ENABLE_PERFORMANCE_TEST
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
    {
        std::array dists{300.f, 150.f};
        auto dAng = 2.f;
        for (float dist : dists) {
            camera.SetPos(glm::vec3{.5f, .5f, .5f} * voxSpacing * glm::vec3{newVoxPerVol} +
                          dist * glm::vec3{0.f, 0.f, 1.f});
            cam2CntrDist =
                glm::distance(camera.GetPos(), .5f * voxSpacing * glm::vec3{newVoxPerVol});

            auto ang = 0.f;
            auto totCostInMs = 0.f;
            uint32_t cnt = 0;
            while (ang <= 360.f) {
                camera.Revolve(cam2CntrDist, -dAng, 0.f);

                const auto &[R, F, U, P] = camera.GetRFUP();
                auto tr = glm::scale(glm::identity<glm::mat4>(), invVoxSpacing);
                tr = tr * glm::mat4{R.x,  R.y,  R.z,  0.f, U.x, U.y, U.z, 0.f,
                                    -F.x, -F.y, -F.z, 0.f, P.x, P.y, P.z, 1.f};
                render(camUnProj, tr, rndrTarget, costInMs);
                totCostInMs += costInMs;

                glViewport(0, 0, rndr.res.x, rndr.res.y);
                glClear(GL_COLOR_BUFFER_BIT);

                glBindTexture(GL_TEXTURE_2D, rndr.texID);
                glBindVertexArray(glRes.VAO);
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const void *)0);
                glBindVertexArray(0);
                glBindTexture(GL_TEXTURE_2D, 0);

                glfwPollEvents();
                glfwSwapBuffers(window);

                ang += dAng;
                ++cnt;
            }

            std::cout << std::format("Revolving at distance {} costs {} ms (avg {} ms, avg {} fps)",
                                     dist, totCostInMs, totCostInMs / cnt,
                                     1000.f * cnt / totCostInMs)
                      << std::endl;
        }
    }
#endif // ENABLE_PERFORMANCE_TEST

TERMINAL:
    ImPlot::DestroyContext();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    release();
    if (rndr.texID != 0)
        glDeleteTextures(1, &rndr.texID);

    releaseGLRes(glRes);
    glfwTerminate();

    return 0;
}
