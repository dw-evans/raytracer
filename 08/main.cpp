#include <SDL2/SDL.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <algorithm>
// #include <GL/gl.h> // I am using this for printing the version information
#include <GL/glew.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <chrono>

using namespace glm;

void printMat4(const mat4 &matrix)
{
    // Prints a mat4
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }
}

template <typename VectorType>
void printvector(const VectorType &vec)
{
    std::cout << "Vector: ";
    for (int i = 0; i < VectorType::length(); ++i)
    {
        std::cout << vec[i];
        if (i < VectorType::length() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

struct Sphere
{
    vec3 pos = vec3(0.0f);
    float r = 1;
};

class Camera
{
public:
    vec3 pos = vec3(0.0f);
    mat4 orientation = mat4(1.0f);
    float fov = 28.072f;
    // float aspect = 16.0f / 9.0f;
    float aspect = 2.0f;
    float nearPlane = 240.0f;
    // float farPlane = 100.0f;

    Camera() {}

    Camera(const vec3 &pos, const mat4 &orient, float fov, float aspect)
        : pos(pos), orientation(orient), fov(fov), aspect(aspect) {}

    mat4 getViewMatrix() const
    {
        return inverse(orientation) * translate(mat4(1.0f), -pos);
    }

    mat4 getProjectionMatrix() const
    {
        return mat4(1.0f);
        // return perspective(radians(fov), aspect, nearPlane, farPlane);
    }

    // mat4 getProjectionMatrix() const
    // {
    //     // Construct orthographic projection matrix for isometric projection
    //     float halfWidth = 10.0f; // Adjust as needed based on your scene
    //     float halfHeight = halfWidth / aspect;
    //     return ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, nearPlane, farPlane);
    // }
};

struct Scene
{
    std::vector<Sphere> objs;
    // std::vector<Light> lights;
};

struct Context
{
    int width = 640;
    int height = 320;
    std::string title = "Dan's window";
    SDL_Window *graphicsApplicationWindow = nullptr;
    SDL_GLContext openGLContext = nullptr;
    bool quit = false;

    GLuint shaderProgram;

    unsigned int vbo;
    unsigned int vao;
    unsigned int ebo;

    Scene scene;
    Camera cam;

    std::chrono::_V2::system_clock::time_point startTime;
};

void CheckGLErrors()
{
    GLenum error;
    std::cerr << "Checking for OpenGL errors" << std::endl;
    while ((error = glGetError()) != GL_NO_ERROR)
    {
        switch (error)
        {
        case GL_INVALID_ENUM:
            std::cerr << "OpenGL Error: GL_INVALID_ENUM" << std::endl;
            break;
        case GL_INVALID_VALUE:
            std::cerr << "OpenGL Error: GL_INVALID_VALUE" << std::endl;
            break;
        case GL_INVALID_OPERATION:
            std::cerr << "OpenGL Error: GL_INVALID_OPERATION" << std::endl;
            break;
        case GL_INVALID_FRAMEBUFFER_OPERATION:
            std::cerr << "OpenGL Error: GL_INVALID_FRAMEBUFFER_OPERATION" << std::endl;
            break;
        case GL_OUT_OF_MEMORY:
            std::cerr << "OpenGL Error: GL_OUT_OF_MEMORY" << std::endl;
            break;
        default:
            std::cerr << "OpenGL Error: Unknown error" << std::endl;
            break;
        }
    }
}

void InitialiseScene(Context &ctx)
{
    // create the sphere
    ctx.scene.objs.resize(1);
    auto sphere = ctx.scene.objs[0];
    sphere.pos = vec3(0.0f, 0.0, -2.0f);

    // create the camera
    auto cam = ctx.cam;

    auto shader = ctx.shaderProgram;

    GLint posViewParams;
    posViewParams = glGetUniformLocation(shader, "ViewParams");

    GLint posCamLocalToWorldMatrix;
    posCamLocalToWorldMatrix = glGetUniformLocation(shader, "CamLocalToWorldMatrix");

    GLint posCamGlobalPos;
    posCamGlobalPos = glGetUniformLocation(shader, "CamGlobalPos");

    GLint posTime;
    posTime = glGetUniformLocation(shader, "time");

    float planeHeight = cam.nearPlane * tan(radians(cam.fov) * 0.5f) * 2.0f;
    float planeWidth = planeHeight * cam.aspect;
    vec3 viewParams(planeWidth, planeHeight, cam.nearPlane);

    auto camGlobalPos = cam.pos;
    auto camLocalToWorldMatrix = cam.getViewMatrix();

    printvector(viewParams);

    glUseProgram(shader);

    glUniform3f(posViewParams, viewParams.x, viewParams.y, viewParams.z);
    glUniformMatrix4fv(posCamLocalToWorldMatrix, 1, GL_FALSE, value_ptr(camLocalToWorldMatrix));
    glUniform3f(posCamGlobalPos, camGlobalPos.x, camGlobalPos.y, camGlobalPos.z);

    CheckGLErrors();
}

void GetOpenGLVersionInfo()
{
    std::cout << "Displaying OpenGL version information:" << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "Shading Language: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
}

GLuint CompileShader(GLenum type, const std::string &source)
{
    GLuint shader = glCreateShader(type);
    const char *src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);

    // dont think this section is the issue
    // const GLchar *p[1];
    // p[0] = src;

    // GLint lengths[1];
    // lengths[0] = strlen(src);
    // glShaderSource(shader, 1, p, lengths);

    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        GLint length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        char *message = (char *)alloca(length * sizeof(char));
        glGetShaderInfoLog(shader, length, &length, message);
        std::cout << "Failed to compile shader: " << message << std::endl;
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint CreateShaderProgram(const std::string &vertexShaderSource, const std::string &fragmentShaderSource)
{
    // Creates the shader program from the vertex and frag shaders

    GLuint program = glCreateProgram();
    // std::cout << vertexShaderSource << std::endl
    //           << std::endl
    //           << std::endl;
    GLuint vs = CompileShader(GL_VERTEX_SHADER, vertexShaderSource);
    // std::cout << fragmentShaderSource << std::endl
    //           << std::endl
    //           << std::endl;
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    glAttachShader(program, vs);
    glAttachShader(program, fs);

    GLint success = 0;
    GLchar errorLog[1024] = {0};

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    std::cout << success << std::endl;
    if (!success)
    {
        glGetProgramInfoLog(program, sizeof(errorLog), NULL, errorLog);
        std::cout << "Error linking shader program: " << stderr << std::endl
                  << errorLog << std::endl;
        exit(1);
    }

    glValidateProgram(program);
    glGetProgramiv(program, GL_VALIDATE_STATUS, &success);
    std::cout << success << std::endl;
    if (!success)
    {
        glGetProgramInfoLog(program, sizeof(errorLog), NULL, errorLog);
        std::cout << "Invalid shader program: " << stderr << std::endl
                  << errorLog << std::endl;
        exit(1);
    }

    // Clean up intermediate shaders
    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

std::string readShaderFile(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file)
    {
        std::cerr << "Failed to open shader file: " << filepath << std::endl;
        return "";
    }

    std::string source;
    std::string line;
    while (std::getline(file, line))
        source += line + '\n';

    return source;
}

void InitialiseProgram(Context &ctx)
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        std::cout << "SDL2 could not initialise video subsystem: " << SDL_GetError() << std::endl;
        exit(1);
    }

    //  create the sdl windo
    ctx.graphicsApplicationWindow = SDL_CreateWindow(
        ctx.title.c_str(),
        0.0,
        0.0,
        ctx.width,
        ctx.height,
        SDL_WINDOW_OPENGL);

    if (ctx.graphicsApplicationWindow == nullptr)
    {
        std::cout << "SDL_Window was not able to be created" << std::endl;
        exit(1);
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    //  create the opengl ctx
    ctx.openGLContext = SDL_GL_CreateContext(ctx.graphicsApplicationWindow);

    if (ctx.openGLContext == nullptr)
    {
        std::cout << "OpenGl ctx was not able to be created" << std::endl;
        exit(1);
    }

    // init glew and do some checks
    GLenum glewError = glewInit();
    if (glewError != GLEW_OK)
    {
        std::cout << "GLEW initialization failed: " << glewGetErrorString(glewError) << std::endl;
        exit(1);
    }

    if (!GLEW_VERSION_3_0)
    {
        std::cout << "OpenGL 3.0 or higher is not supported." << std::endl;
        exit(1);
    }

    // Display the version info
    // skip while we use glew
    GetOpenGLVersionInfo();

    // create the shaders

    std::string vertexShaderSource = readShaderFile("shaders/vs.glsl");
    std::string fragmentShaderSource = readShaderFile("shaders/fs.glsl");

    ctx.shaderProgram = CreateShaderProgram(
        vertexShaderSource,
        fragmentShaderSource);

    float vertices[] = {
        1.0f, 1.0f, 0.0f,   // top right
        1.0f, -1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, // bottom left
        -1.0f, 1.0f, 0.0f   // top left
    };

    // // double the vertex values to get the whole screen
    // for (int i = 0; i < 16; ++i)
    // {
    //     vertices[i] = vertices[i] * 1.5;
    // }

    // int numVals = sizeof(vertices) / sizeof(vertices[0]);
    // //  hack scale them by two to fill the whole screen
    // for (int i; i < numVals; ++i)
    // {
    //     // if (i % 3 == 0) {
    //     vertices[i] = vertices[i] * 2;
    //     // }
    // }

    unsigned int indices[] = {
        // note that we start from 0!
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    glGenVertexArrays(1, &ctx.vao);
    glGenBuffers(1, &ctx.vbo);

    glBindVertexArray(ctx.vao);
    glBindBuffer(GL_ARRAY_BUFFER, ctx.vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    // copy the index array into an element buffer and set attributes pointers
    glGenBuffers(1, &ctx.ebo);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ctx.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo2);
    // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices2), indices2, GL_STATIC_DRAW);

    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // ctx.startTime = std::chrono::high_resolution_clock::now()

    InitialiseScene(ctx);
}

void Input(Context &ctx)
{
    SDL_Event e;
    while (SDL_PollEvent(&e) != 0)
    {
        if (e.type == SDL_QUIT)
        {
            std::cout << "Goodbye" << std::endl;
            ctx.quit = true;
        }
    }
}

void PreDraw(Context &ctx)
{
}

void Draw(Context &ctx)
{
}

double timeValue;

void MainLoop(Context &ctx)
{
    while (!ctx.quit)
    {

        Input(ctx);

        // Draw(ctx);
        glUseProgram(ctx.shaderProgram);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // glClear(GL_COLOR_BUFFER_BIT);

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::microseconds>(ctx.startTime).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(currentTime).time_since_epoch().count();
        float ftime = (end - start) * 0.001f * 0.001f;
        // std::cout << "time is: " << ftime << std::endl;

        GLint posTime;
        posTime = glGetUniformLocation(ctx.shaderProgram, "time");
        glUniform1f(posTime, ftime);

        glBindVertexArray(ctx.vao);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ctx.ebo);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // update the screen
        SDL_GL_SwapWindow(ctx.graphicsApplicationWindow);
    }
}

void CleanUp(Context &ctx)
{
    SDL_DestroyWindow(ctx.graphicsApplicationWindow);
    SDL_Quit();
}

int main()
{

    Context ctx;

    ctx.startTime = std::chrono::high_resolution_clock::now();

    InitialiseProgram(ctx);

    MainLoop(ctx);

    CleanUp(ctx);

    return 0;
}