#include <string_view>

constexpr std::string_view QuadVSCode = "#version 450 core\n"

                                        "layout(location = 0) in vec2 positionIn;\n"
                                        "layout(location = 1) in vec2 uvIn;\n"

                                        "out vec2 uv;\n"

                                        "void main() {\n"
                                        "    gl_Position = vec4(positionIn, 0, 1.0);\n"
                                        "    uv = uvIn;\n"
                                        "}\n";
