#include <string_view>

constexpr std::string_view QuadFSCode = "#version 450 core\n"

                                        "uniform sampler2D tex;\n"

                                        "in vec2 uv;\n"

                                        "out vec4 fragColor;\n"

                                        "void main() {\n"
                                        "	fragColor = texture(tex, uv);\n"
                                        "}\n";