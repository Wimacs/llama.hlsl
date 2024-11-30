#pragma target sm_6_0

#define MATRIX_WIDTH 1024
#define MATRIX_HEIGHT 1024

RWStructuredBuffer<float> matrixA : register(u0);
RWStructuredBuffer<float> matrixB : register(u1);
RWStructuredBuffer<float> result : register(u2);


StructuredBuffer<float> token_embedding_table : register(t0);
StructuredBuffer<float> rms_att_weight : register(t1);
StructuredBuffer<float> rms_ffn_weight : register(t2);
StructuredBuffer<float> wq : register(t3);
StructuredBuffer<float> wk : register(t4);
StructuredBuffer<float> wv : register(t5);
StructuredBuffer<float> wo : register(t6);
StructuredBuffer<float> w1 : register(t7);
StructuredBuffer<float> w2 : register(t8);
StructuredBuffer<float> w3 : register(t9);
StructuredBuffer<float> rms_final_weight : register(t10);
StructuredBuffer<float> wcls : register(t11);



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
    
    result[DTid.y * MATRIX_WIDTH + DTid.x] = rms_final_weight[DTid.x];
}