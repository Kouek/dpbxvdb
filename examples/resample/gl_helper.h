#ifndef KOUEK_GL_HELPER_H
#define KOUEK_GL_HELPER_H

#include <memory>

#include <glad/glad.h>

#include "shader.h"

#include "quad.vs"
#include "quad.fs"

#define GLCheck                                                                                    \
    {                                                                                              \
        GLenum glErr;                                                                              \
        if ((glErr = glGetError()) != GL_NO_ERROR) {                                               \
            std::cerr << "OpenGL err: " << glErr << " caused before on line " << __LINE__          \
                      << "  of file:" << __FILE__ << std::endl;                                    \
        }                                                                                          \
    }

struct GLResource {
    GLuint VAO = 0;
    GLuint VBO = 0;
    GLuint EBO = 0;
    std::shared_ptr<Shader> shader;
};

inline GLResource genGLRes() {
    GLuint VBO = 0, VAO = 0, EBO = 0;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    {
        std::vector<GLfloat> verts = {-1.f, -1.f, 0, 0,   1.f, -1.f, 1.f, 0,
                                      -1.f, 1.f,  0, 1.f, 1.f, 1.f,  1.f, 1.f};
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * verts.size(), verts.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4, (const void *)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4,
                              (const void *)(sizeof(GLfloat) * 2));
    }
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    {
        GLushort idxes[] = {0, 1, 3, 0, 3, 2};
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * 6, idxes, GL_STATIC_DRAW);
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    std::vector<const char *> uniforms;
    auto shader = std::make_shared<Shader>(uniforms, QuadVSCode.data(), QuadFSCode.data());

    return GLResource{VAO, VBO, EBO, shader};
}

inline void releaseGLRes(GLResource &res) {
    glDeleteVertexArrays(1, &res.VAO);
    glDeleteBuffers(1, &res.VBO);
    glDeleteBuffers(1, &res.EBO);
    res.VAO = res.VBO = res.EBO = 0;

    res.shader.reset();
}

#endif // !KOUEK_GL_HELPER_H
