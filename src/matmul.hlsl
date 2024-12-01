#pragma target sm_6_0
RWStructuredBuffer<float> x : register(u0);
RWStructuredBuffer<float> xb : register(u1);
RWStructuredBuffer<float> xb2 : register(u2);
RWStructuredBuffer<float> hb : register(u3);
RWStructuredBuffer<float> hb2 : register(u4);
RWStructuredBuffer<float> q : register(u5);
RWStructuredBuffer<float> k : register(u6);
RWStructuredBuffer<float> v : register(u7);
RWStructuredBuffer<float> att : register(u8);
RWStructuredBuffer<float> logits : register(u9);
RWStructuredBuffer<float> key_cache : register(u10);
RWStructuredBuffer<float> value_cache : register(u11);

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

[numthreads(32, 1, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID)
{
    logits[DTid.x] = token_embedding_table[DTid.x];
}