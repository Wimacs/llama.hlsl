//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#pragma once

#include <corecrt_io.h>
#include <fcntl.h>

#include "win.h"
#include "../DXSample/DXSample.h"
#include "../DXSample/SimpleCamera.h"
#include "../DXSample/StepTimer.h"

using namespace DirectX;

// Note that while ComPtr is used to manage the lifetime of resources on the CPU,
// it has no understanding of the lifetime of resources on the GPU. Apps must account
// for the GPU lifetime of resources to avoid destroying objects that may still be
// referenced by the GPU.
// An example of this can be found in the class method: OnDestroy().
using Microsoft::WRL::ComPtr;

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float* x; // activation at current time stamp (dim,)
    float* xb; // same, but inside a residual branch (dim,)
    float* xb2; // an additional buffer just for convenience (dim,)
    float* hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float* q; // query (dim,)
    float* k; // key (dim,)
    float* v; // value (dim,)
    float* att; // buffer for scores/attention values (n_heads, seq_len)
    float* logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

inline void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

inline void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

inline void memory_map_weights(TransformerWeights* w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

inline void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
    int* fd, float** data, ssize_t* file_size) {
    FILE* file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config) / sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

class D3D12llm : public DXSample
{
public:
    D3D12llm(UINT width, UINT height, std::wstring name, Transformer* transformer);

    virtual void OnInit();
    virtual void OnUpdate();
    virtual void OnRender();
    virtual void OnDestroy();
    virtual void OnKeyDown(UINT8 key);
    virtual void OnKeyUp(UINT8 key);

private:
    static const UINT FrameCount = 2;
    static const UINT ThreadCount = 5;
    static const float ParticleSpread;
    static const UINT ParticleCount = 100000;        // The number of particles in the n-body simulation.

    // "Vertex" definition for particles. Triangle vertices are generated 
    // by the geometry shader. Color data will be assigned to those 
    // vertices via this struct.
    struct ParticleVertex
    {
        XMFLOAT4 color;
    };

    // Position and velocity data for the particles in the system.
    // Two buffers full of Particle data are utilized in this sample.
    // The compute thread alternates writing to each of them.
    // The render thread renders using the buffer that is not currently
    // in use by the compute shader.
    struct Particle
    {
        XMFLOAT4 position;
        XMFLOAT4 velocity;
    };

    struct ConstantBufferGS
    {
        XMFLOAT4X4 worldViewProjection;
        XMFLOAT4X4 inverseView;

        // Constant buffers are 256-byte aligned in GPU memory. Padding is added
        // for convenience when computing the struct's size.
        float padding[32];
    };

    struct ConstantBufferCS
    {
        UINT param[4];
        float paramf[4];
    };


    //llama data
    Transformer* tansformer;

    // Pipeline objects.
    CD3DX12_VIEWPORT m_viewport;
    CD3DX12_RECT m_scissorRect;
    ComPtr<IDXGISwapChain3> m_swapChain;
    ComPtr<ID3D12Device> m_device;
    ComPtr<ID3D12Resource> m_renderTargets[FrameCount];
    UINT m_frameIndex;
    ComPtr<ID3D12CommandAllocator> m_commandAllocators[FrameCount];
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12RootSignature> m_computeRootSignature;
    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    UINT m_rtvDescriptorSize;
    UINT m_srvUavDescriptorSize;

    ComPtr<ID3D12DescriptorHeap> m_imGuiSrvDescHeap;
    bool m_showImGuiDemo = true;

    // Asset objects.
    ComPtr<ID3D12PipelineState> m_pipelineState;
    ComPtr<ID3D12PipelineState> m_computeState;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12Resource> m_vertexBuffer;
    ComPtr<ID3D12Resource> m_vertexBufferUpload;
    D3D12_VERTEX_BUFFER_VIEW m_vertexBufferView;
    ComPtr<ID3D12Resource> m_particleBuffer0[ThreadCount];
    ComPtr<ID3D12Resource> m_particleBuffer1[ThreadCount];
    ComPtr<ID3D12Resource> m_particleBuffer0Upload[ThreadCount];
    ComPtr<ID3D12Resource> m_particleBuffer1Upload[ThreadCount];
    ComPtr<ID3D12Resource> m_constantBufferGS;
    UINT8* m_pConstantBufferGSData;
    ComPtr<ID3D12Resource> m_constantBufferCS;

    UINT m_srvIndex[ThreadCount];        // Denotes which of the particle buffer resource views is the SRV (0 or 1). The UAV is 1 - srvIndex.
    UINT m_heightInstances;
    UINT m_widthInstances;
    SimpleCamera m_camera;
    StepTimer m_timer;

    // Compute objects.
    ComPtr<ID3D12CommandAllocator> m_computeAllocator[ThreadCount];
    ComPtr<ID3D12CommandQueue> m_computeCommandQueue[ThreadCount];
    ComPtr<ID3D12GraphicsCommandList> m_computeCommandList[ThreadCount];

    // Synchronization objects.
    HANDLE m_swapChainEvent;
    ComPtr<ID3D12Fence> m_renderContextFence;
    UINT64 m_renderContextFenceValue;
    HANDLE m_renderContextFenceEvent;
    UINT64 m_frameFenceValues[FrameCount];

    ComPtr<ID3D12Fence> m_threadFences[ThreadCount];
    volatile HANDLE m_threadFenceEvents[ThreadCount];

    // Thread state.
    LONG volatile m_terminating;
    UINT64 volatile m_renderContextFenceValues[ThreadCount];
    UINT64 volatile m_threadFenceValues[ThreadCount];

    struct ThreadData
    {
        D3D12llm* pContext;
        UINT threadIndex;
    };
    ThreadData m_threadData[ThreadCount];
    HANDLE m_threadHandles[ThreadCount];

    // Indices of the root signature parameters.
    enum GraphicsRootParameters : UINT32
    {
        GraphicsRootCBV = 0,
        GraphicsRootSRVTable,
        GraphicsRootParametersCount
    };

    enum ComputeRootParameters : UINT32
    {
        ComputeRootCBV = 0,
        ComputeRootSRVTable,
        ComputeRootUAVTable,
        ComputeRootParametersCount
    };

    // Indices of shader resources in the descriptor heap.
    enum DescriptorHeapIndex : UINT32
    {
        UavParticlePosVelo0 = 0,
        UavParticlePosVelo1 = UavParticlePosVelo0 + ThreadCount,
        SrvParticlePosVelo0 = UavParticlePosVelo1 + ThreadCount,
        SrvParticlePosVelo1 = SrvParticlePosVelo0 + ThreadCount,
        DescriptorCount = SrvParticlePosVelo1 + ThreadCount
    };

    void LoadPipeline();
    void LoadAssets();
    void RestoreD3DResources();
    void ReleaseD3DResources();
    void WaitForGpu();
    void CreateAsyncContexts();
    void CreateVertexBuffer();
    float RandomPercent();
    void LoadParticles(_Out_writes_(numParticles) Particle* pParticles, const XMFLOAT3 &center, const XMFLOAT4 &velocity, float spread, UINT numParticles);
    void CreateParticleBuffers();
    void PopulateCommandList();
    void InitImGui();
    void RenderImGui();

    static DWORD WINAPI ThreadProc(ThreadData* pData)
    {
        return pData->pContext->AsyncComputeThreadProc(pData->threadIndex);
    }
    DWORD AsyncComputeThreadProc(int threadIndex);
    void Simulate(UINT threadIndex);

    void WaitForRenderContext();
    void MoveToNextFrame();
};
