#include "shared.h"

// =========================================================================
//   Resources
// =========================================================================

// Constant buffer with data needed for NN
cbuffer NNDataCB : register(b0)
{
    NNData gData;
}


// Network weights and biases
RWStructuredBuffer<float> nnWeights : register(u1);
RWStructuredBuffer<float> nnBiases : register(u2);



[numthreads(8, 8, 1)]
void Inference(
	int2 groupID : SV_GroupID,
	int2 groupThreadID : SV_GroupThreadID,
	int2 LaunchIndex : SV_DispatchThreadID)
{

}



[numthreads(16, 16, 1)]
void Initialize(
	int2 groupID : SV_GroupID,
	int2 groupThreadID : SV_GroupThreadID,
	int2 LaunchIndex : SV_DispatchThreadID)
{

}

