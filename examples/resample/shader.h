#ifndef KOUEK_SHADER_H
#define KOUEK_SHADER_H

#include <string>
#include <vector>

#include <glad/glad.h>

struct Shader {
    bool ok = false;
    GLuint prog;

    std::vector<const char *> uniformNames;
    std::vector<GLint> uniformLocations;
    std::string errMsg;

    Shader(const std::vector<const char *> &uniforms, const char *vsCode, const char *fsCode) {
        GLuint vs;
        GLuint fs;
        vs = glCreateShader(GL_VERTEX_SHADER);
        fs = glCreateShader(GL_FRAGMENT_SHADER);
        prog = glCreateProgram();

        auto checkCompilation = [&](GLuint ID, const char *type) {
            GLint success;
            GLchar errInfo[1024];

            if (type != "PROG")
                glGetShaderiv(ID, GL_COMPILE_STATUS, &success);
            else
                glGetProgramiv(ID, GL_LINK_STATUS, &success);
            if (success)
                return true;

            if (type != "PROG")
                glGetShaderInfoLog(ID, sizeof(errInfo), nullptr, errInfo);
            else
                glGetProgramInfoLog(ID, sizeof(errInfo), nullptr, errInfo);
            errMsg.assign(type);
            errMsg.append(" err: ");
            errMsg.append(errInfo);
            return false;
        };

        glShaderSource(vs, 1, &vsCode, nullptr);
        glCompileShader(vs);
        if (!checkCompilation(vs, "VS"))
            goto TERMINAL;

        glShaderSource(fs, 1, &fsCode, nullptr);
        glCompileShader(fs);
        if (!checkCompilation(fs, "FS"))
            goto TERMINAL;

        glAttachShader(prog, vs);
        glAttachShader(prog, fs);
        glLinkProgram(prog);
        if (!checkCompilation(prog, "PROG"))
            goto TERMINAL;

        uniformNames = uniforms;
        uniformLocations.reserve(uniformNames.size());
        for (const auto name : uniformNames) {
            GLint location = glGetUniformLocation(prog, name);
            if (location == -1) {
                errMsg += "Cannot find uniform: ";
                errMsg += name;
                goto TERMINAL;
            }
            uniformLocations.emplace_back(location);
        }

        ok = true;
    TERMINAL:
        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    ~Shader() { glDeleteProgram(prog); }
};

#endif // !KOUEK_SHADER_H
