#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

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

mat4 mat4fromvec4(vec4 &v, int idx, bool rowWise = false)
{
    // creates a mat4 from a vec4
    mat4 ret = mat4(0.0f);

    // do it row-wise
    if (rowWise)
    {
        for (int i = 0; i < 4; ++i)
        {
            ret[idx][i] = v[i];
        }
    }
    // do it column-wise
    else
    {
        for (int i = 0; i < 4; ++i)
        {
            ret[i][idx] = v[i];
        }
    }

    return ret;
}

int main()
{

    mat4 mymatrix = mat4(1.0f);

    // printMat4(mymatrix);

    vec3 myvec = vec3(2.0f);

    printvector(normalize(myvec));
    myvec.x = 10.0f;
    myvec.y = -5.0f;
    myvec.z = 2.0f;

    printvector(normalize(myvec));

    mymatrix = translate(mymatrix, myvec);

    // printMat4(mymatrix);

    return 0;
}