#pragma target sm_6_0

#define MATRIX_WIDTH 1024
#define MATRIX_HEIGHT 1024

RWStructuredBuffer<float> matrixA : register(u0);
RWStructuredBuffer<float> matrixB : register(u1);
RWStructuredBuffer<float> result : register(u2);

[numthreads(32, 32, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= MATRIX_WIDTH || DTid.y >= MATRIX_HEIGHT)
        return;

    float sum = 0.0f;
    for (uint k = 0; k < MATRIX_WIDTH; k++)
    {
        sum += matrixA[DTid.y * MATRIX_WIDTH + k] * matrixB[k * MATRIX_WIDTH + DTid.x];
    }
    
    result[DTid.y * MATRIX_WIDTH + DTid.x] = sum;
}