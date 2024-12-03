#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include "d3dx12.h"
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <iostream>
#include <wrl.h>
#include <vector>
#include <winrt/base.h>
#include <guiddef.h>
#include <unknwn.h>
#include <winrt/base.h>
#include <initguid.h>
#include <dxcore.h>
#include <dxcapi.h>

using namespace Microsoft::WRL;

// DX12 related variables
ComPtr<ID3D12Device> device;
ComPtr<ID3D12CommandQueue> commandQueue;
ComPtr<ID3D12CommandAllocator> commandAllocator;
ComPtr<ID3D12GraphicsCommandList> commandList;
ComPtr<ID3D12RootSignature> rootSignature;
ComPtr<ID3D12PipelineState> pipelineState;


// Synchronization related variables
ComPtr<ID3D12Fence> fence;
UINT64 fenceValue = 0;
HANDLE fenceEvent;

// Add in global variable area ...
ComPtr<ID3D12DescriptorHeap> computeHeap;
UINT descriptorSize;


typedef struct {
    // token embedding table
    ComPtr<ID3D12Resource> token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    ComPtr<ID3D12Resource> rms_att_weight; // (layer, dim) rmsnorm weights
    ComPtr<ID3D12Resource> rms_att_weight_upload;
    ComPtr<ID3D12Resource> rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    ComPtr<ID3D12Resource> wq; // (layer, dim, n_heads * head_size)
    ComPtr<ID3D12Resource> wk; // (layer, dim, n_kv_heads * head_size)
    ComPtr<ID3D12Resource> wv; // (layer, dim, n_kv_heads * head_size)
    ComPtr<ID3D12Resource> wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    ComPtr<ID3D12Resource> w1; // (layer, hidden_dim, dim)
    ComPtr<ID3D12Resource> w2; // (layer, dim, hidden_dim)
    ComPtr<ID3D12Resource> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    ComPtr<ID3D12Resource> rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    ComPtr<ID3D12Resource> wcls;
    ComPtr<ID3D12Resource> token_embedding_table_upload;
    ComPtr<ID3D12Resource> rms_ffn_weight_upload;
    ComPtr<ID3D12Resource> wq_upload;
    ComPtr<ID3D12Resource> wk_upload;
    ComPtr<ID3D12Resource> wv_upload;
    ComPtr<ID3D12Resource> wo_upload;
    ComPtr<ID3D12Resource> w1_upload;
    ComPtr<ID3D12Resource> w2_upload;
    ComPtr<ID3D12Resource> w3_upload;
    ComPtr<ID3D12Resource> rms_final_weight_upload;
    ComPtr<ID3D12Resource> wcls_upload;
} TransformerWeightsGPU;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    size_t token_embedding_table_size;
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    size_t rms_att_weight_size;
    float* rms_ffn_weight; // (layer, dim)
    size_t rms_ffn_weight_size;
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    size_t wq_size;
    float* wk; // (layer, dim, n_kv_heads * head_size)
    size_t wk_size;
    float* wv; // (layer, dim, n_kv_heads * head_size)
    size_t wv_size;
    float* wo; // (layer, n_heads * head_size, dim)
    size_t wo_size;
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    size_t w1_size;
    float* w2; // (layer, dim, hidden_dim)
    size_t w2_size;
    float* w3; // (layer, hidden_dim, dim)
    size_t w3_size;
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    size_t rms_final_weight_size;
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
    size_t wcls_size;
} TransformerWeights;


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
    // current wave of activations
    ComPtr<ID3D12Resource> x;          // activation at current time stamp (dim,)
    size_t x_size;                     // dim
    ComPtr<ID3D12Resource> xb;         // same, but inside a residual branch (dim,)
    size_t xb_size;                    // dim
    ComPtr<ID3D12Resource> xb2;        // additional buffer for convenience (dim,)
    size_t xb2_size;                   // dim
    ComPtr<ID3D12Resource> hb;         // buffer for hidden dimension in the ffn (hidden_dim,)
    size_t hb_size;                    // hidden_dim
    ComPtr<ID3D12Resource> hb2;        // buffer for hidden dimension in the ffn (hidden_dim,)
    size_t hb2_size;                   // hidden_dim
    ComPtr<ID3D12Resource> q;          // query (dim,)
    size_t q_size;                     // dim
    ComPtr<ID3D12Resource> k;          // key (dim,)
    size_t k_size;                     // dim
    ComPtr<ID3D12Resource> v;          // value (dim,)
    size_t v_size;                     // dim
    ComPtr<ID3D12Resource> att;        // buffer for scores/attention values (n_heads, seq_len)
    size_t att_size;                   // n_heads * seq_len
    ComPtr<ID3D12Resource> logits;     // output logits (vocab_size,)
    ComPtr<ID3D12Resource> logits_readback;
    size_t logits_size;                // vocab_size

    // kv cache
    ComPtr<ID3D12Resource> key_cache;   // (layer, seq_len, dim)
    size_t key_cache_size;              // n_layers * seq_len * dim
    ComPtr<ID3D12Resource> value_cache; // (layer, seq_len, dim)
    size_t value_cache_size;            // n_layers * seq_len * dim
} RunStateGPU;


typedef struct {
    Config config;
    TransformerWeights weights;
    TransformerWeightsGPU weights_gpu;
    RunStateGPU runstate_gpu;
    // Windows specific handle
    HANDLE file_handle;
    float* data;
    size_t file_size;
} Transformer;


void WaitForGpu() {
    const UINT64 currentFenceValue = fenceValue;
    commandQueue->Signal(fence.Get(), currentFenceValue);
    fenceValue++;

    if (fence->GetCompletedValue() < currentFenceValue) {
        fence->SetEventOnCompletion(currentFenceValue, fenceEvent);
        WaitForSingleObject(fenceEvent, INFINITE);
    }
}


bool TryGetProperty(IDXCoreAdapter* adapter, DXCoreAdapterProperty prop, std::string& outputValue)
{
    if (adapter->IsPropertySupported(prop))
    {
        size_t propSize;
        (adapter->GetPropertySize(prop, &propSize));

        outputValue.resize(propSize);
        (adapter->GetProperty(prop, propSize, outputValue.data()));

        // Trim any trailing nul characters. 
        while (!outputValue.empty() && outputValue.back() == '\0')
        {
            outputValue.pop_back();
        }

        return true;
    }
    return false;
}

ComPtr<IDXCoreAdapter> GetAdapter() {
    ComPtr<IDXCoreAdapterFactory> factory;
    if (FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory)))) {
        throw std::runtime_error("Failed to create DXCore factory");
    }

    ComPtr<IDXCoreAdapterList> adapterList;
    if (FAILED(factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE, 
        IID_PPV_ARGS(&adapterList)))) {
        throw std::runtime_error("Failed to create adapter list");
    }

    std::vector<ComPtr<IDXCoreAdapter>> dxCoreAdapters;
    uint32_t adapterCount = adapterList->GetAdapterCount();
    
    for (uint32_t i = 0; i < adapterCount; ++i) {
        ComPtr<IDXCoreAdapter> adapter;
        if (SUCCEEDED(adapterList->GetAdapter(i, IID_PPV_ARGS(&adapter)))) {
            bool isHardware = false;
        	{
                char description[128];
                (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, description)));
                std::cout << i + 1 << ". " << description << std::endl;

                // Get memory information
                uint64_t dedicatedMemory = 0;
                if (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory,
                    &dedicatedMemory))) {
                    std::cout << "mem:" << dedicatedMemory / 1024 / 1024 << " MB" << std::endl;
                    std::string test;
                    if (TryGetProperty(adapter.Get(), DXCoreAdapterProperty::DriverDescription, test))
                        std::cout << test << std::endl;

                    dxCoreAdapters.push_back(adapter);
                }
            }
        }
    }

    if (dxCoreAdapters.empty()) {
        throw std::runtime_error("No compatible GPU device found");
    }

    std::cout << "please choot(1-" << dxCoreAdapters.size() << "): ";
    int choice;
    std::cin >> choice;

    if (choice < 1 || choice > static_cast<int>(dxCoreAdapters.size())) {
        throw std::runtime_error("Invalid adapter selection");
    }

    return dxCoreAdapters[choice - 1];
}

// ... existing code ...


// Modify InitializeDevice function
void InitializeDevice() {
    ComPtr<IDXCoreAdapter> selectedAdapter = GetAdapter();
    // Change D3D_FEATURE_LEVEL_12_2 to explicitly check 12.0 or higher version
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_12_2,
        D3D_FEATURE_LEVEL_12_1,
        D3D_FEATURE_LEVEL_12_0
    };
    
    D3D_FEATURE_LEVEL supportedLevel;
    for (auto level : featureLevels) {
        if (SUCCEEDED(D3D12CreateDevice(selectedAdapter.Get(), level, IID_PPV_ARGS(&device)))) {
            supportedLevel = level;
            break;
        }
    }

    if (!device) {
        throw std::runtime_error("Failed to create D3D12 device with required feature level");
    }

    // Verify shader model 6.0 support
    D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_0 };
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))) 
        || shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_0) {
        throw std::runtime_error("Shader Model 6.0 is not supported");
    }
    
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    if (FAILED(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)))) {
        throw std::runtime_error("Failed to create command queue");
    }
    
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, 
        IID_PPV_ARGS(&commandAllocator)))) {
        throw std::runtime_error("Failed to create command allocator");
    }
    
    if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, 
        commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList)))) {
        throw std::runtime_error("Failed to create command list");
    }
}

// Create compute pipeline
void CreateComputePipeline() {
    // Create CBV/SRV/UAV descriptor heap
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 12 + 12; // 12 UAVs for run state, and 12 for weights
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&computeHeap));
    
    descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create root signature
    CD3DX12_ROOT_PARAMETER rootParameter;
    CD3DX12_DESCRIPTOR_RANGE ranges[2];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 12, 0, 0, 0);//12 uav for runstate
	ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 12, 0,0,12);//12 srv for weights
	// 12SRV, 12 UAVs, starting register 0
    rootParameter.InitAsDescriptorTable(2, ranges);

    CD3DX12_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init(1, &rootParameter, 0, nullptr, 
        D3D12_ROOT_SIGNATURE_FLAG_NONE);
    
    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, 
        &signature, &error);
    device->CreateRootSignature(0, signature->GetBufferPointer(), 
        signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature));

    // Initialize DXC compiler
    ComPtr<IDxcUtils> dxcUtils;
    ComPtr<IDxcCompiler3> dxcCompiler;
    ComPtr<IDxcIncludeHandler> dxcIncludeHandler;
    
    DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils));
    DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler));
    dxcUtils->CreateDefaultIncludeHandler(&dxcIncludeHandler);

    // Load shader file
    ComPtr<IDxcBlobEncoding> shaderSource;
    dxcUtils->LoadFile(L"../src/matmul.hlsl", nullptr, &shaderSource);

    // Set compilation parameters
    LPCWSTR args[] = {
        L"-E", L"CSMain",  // Entry point
        L"-T", L"cs_6_0",  // target profile
        L"-Zi",            // Debug information
        L"-Od"             // Disable optimization
    };

    // Create compilation parameters
    DxcBuffer sourceBuffer;
    sourceBuffer.Ptr = shaderSource->GetBufferPointer();
    sourceBuffer.Size = shaderSource->GetBufferSize();
    sourceBuffer.Encoding = DXC_CP_ACP;

    // Compile shader
    ComPtr<IDxcResult> dxcResult;
    HRESULT hr = dxcCompiler->Compile(
        &sourceBuffer,     // source code
        args,             // arguments
        _countof(args),   // number of arguments
        dxcIncludeHandler.Get(),  // include handler
        IID_PPV_ARGS(&dxcResult)  // compilation result
    );

    // Check compilation result
    ComPtr<IDxcBlobUtf8> errors;
    dxcResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
    if (errors && errors->GetStringLength() > 0) {
        std::cout << "Shader compilation failed with error: " 
                  << errors->GetStringPointer() << std::endl;
    }

    HRESULT status;
    dxcResult->GetStatus(&status);
    if (FAILED(status)) {
        throw std::runtime_error("Failed to compile compute shader");
    }

    // Get compiled shader bytecode
    ComPtr<IDxcBlob> computeShader;
    dxcResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&computeShader), nullptr);

    // Create PSO
    D3D12_COMPUTE_PIPELINE_STATE_DESC computePsoDesc = {};
    computePsoDesc.pRootSignature = rootSignature.Get();
    computePsoDesc.CS = CD3DX12_SHADER_BYTECODE(computeShader->GetBufferPointer(), 
        computeShader->GetBufferSize());
    device->CreateComputePipelineState(&computePsoDesc, IID_PPV_ARGS(&pipelineState));
}

void create_run_state_gpu(RunStateGPU* s, Config* p, ID3D12Device* device, ID3D12DescriptorHeap* uavHeap, CD3DX12_CPU_DESCRIPTOR_HANDLE& handle) {
    // Calculate kv_dim
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    // Set all buffer sizes
    s->x_size = p->dim;
    s->xb_size = p->dim;
    s->xb2_size = p->dim;
    s->hb_size = p->hidden_dim;
    s->hb2_size = p->hidden_dim;
    s->q_size = p->dim;
    s->k_size = p->dim;
    s->v_size = p->dim;
    s->att_size = p->n_heads * p->seq_len;
    s->logits_size = p->vocab_size;
    s->key_cache_size = p->n_layers * p->seq_len * kv_dim;
    s->value_cache_size = p->n_layers * p->seq_len * kv_dim;

    // Create default heap properties (GPU memory)
    auto heapProps = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);

    // Create buffer lambda function
    auto CreateBuffer = [&](size_t numElements, ComPtr<ID3D12Resource>& resource) {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            numElements * sizeof(float),
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS  // Allow UAV access
        );

        if (FAILED(device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,  // Initial state set to UAV
            nullptr,
            IID_PPV_ARGS(&resource)))) {
            throw std::runtime_error("Failed to create buffer resource");
        }
    };

    // Create all buffers
    CreateBuffer(s->x_size, s->x);
    CreateBuffer(s->xb_size, s->xb);
    CreateBuffer(s->xb2_size, s->xb2);
    CreateBuffer(s->hb_size, s->hb);
    CreateBuffer(s->hb2_size, s->hb2);
    CreateBuffer(s->q_size, s->q);
    CreateBuffer(s->k_size, s->k);
    CreateBuffer(s->v_size, s->v);
    CreateBuffer(s->att_size, s->att);
    CreateBuffer(s->logits_size, s->logits);
    CreateBuffer(s->key_cache_size, s->key_cache);
    CreateBuffer(s->value_cache_size, s->value_cache);

    // Get UAV descriptor size
    UINT descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create UAV lambda function
    auto CreateUAV = [&](ID3D12Resource* resource, size_t numElements, CD3DX12_CPU_DESCRIPTOR_HANDLE& handle) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = numElements;
        uavDesc.Buffer.StructureByteStride = 0;
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

        device->CreateUnorderedAccessView(
            resource,
            nullptr,
            &uavDesc,
            handle);

        handle.Offset(descriptorSize);
        };

    // Create all UAVs    
    CreateUAV(s->x.Get(), p->dim, handle);
    CreateUAV(s->xb.Get(), p->dim, handle);
    CreateUAV(s->xb2.Get(), p->dim, handle);
    CreateUAV(s->hb.Get(), p->hidden_dim, handle);
    CreateUAV(s->hb2.Get(), p->hidden_dim, handle);
    CreateUAV(s->q.Get(), p->dim, handle);
    CreateUAV(s->k.Get(), p->dim, handle);
    CreateUAV(s->v.Get(), p->dim, handle);
    CreateUAV(s->att.Get(), p->n_heads * p->seq_len, handle);
    CreateUAV(s->logits.Get(), p->vocab_size, handle);
    CreateUAV(s->key_cache.Get(), p->n_layers * p->seq_len * kv_dim, handle);
    CreateUAV(s->value_cache.Get(), p->n_layers * p->seq_len * kv_dim, handle);
}

// Create resource buffers
void CreateResources(Transformer* transformer) {
    // Create matrix buffers
    auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);


    // Create resources for each weight
    auto CreateWeightResource = [&](size_t size, ComPtr<ID3D12Resource>& resource) {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(
            size * sizeof(float),
            D3D12_RESOURCE_FLAG_NONE
        );
        device->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
            nullptr,
            IID_PPV_ARGS(&resource)
        );
    };

    // Create all weight resources
    CreateWeightResource(transformer->weights.token_embedding_table_size, transformer->weights_gpu.token_embedding_table);
    CreateWeightResource(transformer->weights.rms_att_weight_size, transformer->weights_gpu.rms_att_weight);
    CreateWeightResource(transformer->weights.rms_ffn_weight_size, transformer->weights_gpu.rms_ffn_weight);
    CreateWeightResource(transformer->weights.wq_size, transformer->weights_gpu.wq);
    CreateWeightResource(transformer->weights.wk_size, transformer->weights_gpu.wk);
    CreateWeightResource(transformer->weights.wv_size, transformer->weights_gpu.wv);
    CreateWeightResource(transformer->weights.wo_size, transformer->weights_gpu.wo);
    CreateWeightResource(transformer->weights.w1_size, transformer->weights_gpu.w1);
    CreateWeightResource(transformer->weights.w2_size, transformer->weights_gpu.w2);
    CreateWeightResource(transformer->weights.w3_size, transformer->weights_gpu.w3);
    CreateWeightResource(transformer->weights.rms_final_weight_size, transformer->weights_gpu.rms_final_weight);
    if (!(transformer->config.vocab_size > 0 ? 1 : 0)) {
        CreateWeightResource(transformer->weights.wcls_size, transformer->weights_gpu.wcls);
    }



    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(computeHeap->GetCPUDescriptorHandleForHeapStart());

    create_run_state_gpu(&transformer->runstate_gpu, &transformer->config, device.Get(), computeHeap.Get(), handle);

    // Create all weight SRVs
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.StructureByteStride = sizeof(float);
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

    auto CreateWeightSRV = [&](ComPtr<ID3D12Resource>& resource, size_t size) {
        srvDesc.Buffer.NumElements = size;
        device->CreateShaderResourceView(resource.Get(), &srvDesc, handle);
        handle.Offset(descriptorSize);
    };

    CreateWeightSRV(transformer->weights_gpu.token_embedding_table, transformer->weights.token_embedding_table_size);
    CreateWeightSRV(transformer->weights_gpu.rms_att_weight, transformer->weights.rms_att_weight_size);
    CreateWeightSRV(transformer->weights_gpu.rms_ffn_weight, transformer->weights.rms_ffn_weight_size);
    CreateWeightSRV(transformer->weights_gpu.wq, transformer->weights.wq_size);
    CreateWeightSRV(transformer->weights_gpu.wk, transformer->weights.wk_size);
    CreateWeightSRV(transformer->weights_gpu.wv, transformer->weights.wv_size);
    CreateWeightSRV(transformer->weights_gpu.wo, transformer->weights.wo_size);
    CreateWeightSRV(transformer->weights_gpu.w1, transformer->weights.w1_size);
    CreateWeightSRV(transformer->weights_gpu.w2, transformer->weights.w2_size);
    CreateWeightSRV(transformer->weights_gpu.w3, transformer->weights.w3_size);
    CreateWeightSRV(transformer->weights_gpu.rms_final_weight, transformer->weights.rms_final_weight_size);
    if (!(transformer->config.vocab_size > 0 ? 1 : 0)) {
        CreateWeightSRV(transformer->weights_gpu.wcls, transformer->weights.wcls_size);
    }

}

// Create synchronization objects
void CreateSyncObjects() {
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
}

// Create upload and read back buffers
void CreateUploadAndReadBackBuffers(Transformer* transformer) {
    auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto readbackHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);

    auto readbackDesc = CD3DX12_RESOURCE_DESC::Buffer(
        transformer->runstate_gpu.logits_size * sizeof(float) // logits size is vocabulary size
    );
    device->CreateCommittedResource(
        &readbackHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &readbackDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&transformer->runstate_gpu.logits_readback)
    );

    // Create upload buffers for each weight
    auto CreateUploadBuffer = [&](size_t size, ComPtr<ID3D12Resource>& upload_buffer) {
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(size * sizeof(float));
        device->CreateCommittedResource(
            &uploadHeapProperties,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&upload_buffer)
        );
    };

    CreateUploadBuffer(transformer->weights.token_embedding_table_size, transformer->weights_gpu.token_embedding_table_upload);
    CreateUploadBuffer(transformer->weights.rms_att_weight_size, transformer->weights_gpu.rms_att_weight_upload);
    CreateUploadBuffer(transformer->weights.rms_ffn_weight_size, transformer->weights_gpu.rms_ffn_weight_upload);
    CreateUploadBuffer(transformer->weights.wq_size, transformer->weights_gpu.wq_upload);
    CreateUploadBuffer(transformer->weights.wk_size, transformer->weights_gpu.wk_upload);
    CreateUploadBuffer(transformer->weights.wv_size, transformer->weights_gpu.wv_upload);
    CreateUploadBuffer(transformer->weights.wo_size, transformer->weights_gpu.wo_upload);
    CreateUploadBuffer(transformer->weights.w1_size, transformer->weights_gpu.w1_upload);
    CreateUploadBuffer(transformer->weights.w2_size, transformer->weights_gpu.w2_upload);
    CreateUploadBuffer(transformer->weights.w3_size, transformer->weights_gpu.w3_upload);
    CreateUploadBuffer(transformer->weights.rms_final_weight_size, transformer->weights_gpu.rms_final_weight_upload);
    if (!(transformer->config.vocab_size > 0 ? 1 : 0)) {
        CreateUploadBuffer(transformer->weights.wcls_size, transformer->weights_gpu.wcls_upload);
    }
}

// Upload matrix data to GPU
void UploadTransformerWeights(Transformer* transformer) {
    // Map and copy data to upload buffers

    // Upload all weight data
    auto UploadWeight = [](ComPtr<ID3D12Resource>& upload_buffer, float* data, size_t size) {
        void* mappedData;
        upload_buffer->Map(0, nullptr, &mappedData);
        memcpy(mappedData, data, size * sizeof(float));
        upload_buffer->Unmap(0, nullptr);
    };

    UploadWeight(transformer->weights_gpu.token_embedding_table_upload, transformer->weights.token_embedding_table, transformer->weights.token_embedding_table_size);
    UploadWeight(transformer->weights_gpu.rms_att_weight_upload, transformer->weights.rms_att_weight, transformer->weights.rms_att_weight_size);
    UploadWeight(transformer->weights_gpu.rms_ffn_weight_upload, transformer->weights.rms_ffn_weight, transformer->weights.rms_ffn_weight_size);
    UploadWeight(transformer->weights_gpu.wq_upload, transformer->weights.wq, transformer->weights.wq_size);
    UploadWeight(transformer->weights_gpu.wk_upload, transformer->weights.wk, transformer->weights.wk_size);
    UploadWeight(transformer->weights_gpu.wv_upload, transformer->weights.wv, transformer->weights.wv_size);
    UploadWeight(transformer->weights_gpu.wo_upload, transformer->weights.wo, transformer->weights.wo_size);
    UploadWeight(transformer->weights_gpu.w1_upload, transformer->weights.w1, transformer->weights.w1_size);
    UploadWeight(transformer->weights_gpu.w2_upload, transformer->weights.w2, transformer->weights.w2_size);
    UploadWeight(transformer->weights_gpu.w3_upload, transformer->weights.w3, transformer->weights.w3_size);
    UploadWeight(transformer->weights_gpu.rms_final_weight_upload, transformer->weights.rms_final_weight, transformer->weights.rms_final_weight_size);
    if (!(transformer->config.vocab_size > 0 ? 1 : 0)) {
        UploadWeight(transformer->weights_gpu.wcls_upload, transformer->weights.wcls, transformer->weights.wcls_size);
    }

    // Record copy commands
    commandList->Reset(commandAllocator.Get(), nullptr);

    // Add all resource barriers
    std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
    auto AddBarrier = [&barriers](ComPtr<ID3D12Resource>& resource) {
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
            resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST
        ));
    };

    std::vector<CD3DX12_RESOURCE_BARRIER> barriers2;
    auto AddBarrier2 = [&barriers2](ComPtr<ID3D12Resource>& resource) {
        barriers2.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
            resource.Get(),
            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST
        ));
        };

    AddBarrier2(transformer->weights_gpu.token_embedding_table);
    AddBarrier2(transformer->weights_gpu.rms_att_weight);
    AddBarrier2(transformer->weights_gpu.rms_ffn_weight);
    AddBarrier2(transformer->weights_gpu.wq);
    AddBarrier2(transformer->weights_gpu.wk);
    AddBarrier2(transformer->weights_gpu.wv);
    AddBarrier2(transformer->weights_gpu.wo);
    AddBarrier2(transformer->weights_gpu.w1);
    AddBarrier2(transformer->weights_gpu.w2);
    AddBarrier2(transformer->weights_gpu.w3);
    AddBarrier2(transformer->weights_gpu.rms_final_weight);
    if (!(transformer->config.vocab_size > 0 ? 1 : 0)) {
        AddBarrier2(transformer->weights_gpu.wcls);
    }

    commandList->ResourceBarrier(barriers.size(), barriers.data());

    // Execute all copy operations
    commandList->CopyResource(transformer->weights_gpu.token_embedding_table.Get(), transformer->weights_gpu.token_embedding_table_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.rms_att_weight.Get(), transformer->weights_gpu.rms_att_weight_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.rms_ffn_weight.Get(), transformer->weights_gpu.rms_ffn_weight_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.wq.Get(), transformer->weights_gpu.wq_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.wk.Get(), transformer->weights_gpu.wk_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.wv.Get(), transformer->weights_gpu.wv_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.wo.Get(), transformer->weights_gpu.wo_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.w1.Get(), transformer->weights_gpu.w1_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.w2.Get(), transformer->weights_gpu.w2_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.w3.Get(), transformer->weights_gpu.w3_upload.Get());
    commandList->CopyResource(transformer->weights_gpu.rms_final_weight.Get(), transformer->weights_gpu.rms_final_weight_upload.Get());
    if (!(transformer->config.vocab_size > 0 ? 1 : 0)) {
        commandList->CopyResource(transformer->weights_gpu.wcls.Get(), transformer->weights_gpu.wcls_upload.Get());
    }

    // Create new resource barrier array for transitioning back to UAV state
    std::vector<CD3DX12_RESOURCE_BARRIER> uavBarriers;
    auto AddUAVBarrier = [&uavBarriers](ComPtr<ID3D12Resource>& resource) {
        uavBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
            resource.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        ));
    };

    std::vector<CD3DX12_RESOURCE_BARRIER> srvBarriers;
    auto AddSRVBarrier = [&srvBarriers](ComPtr<ID3D12Resource>& resource) {
        srvBarriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
            resource.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
        ));
        };

    AddSRVBarrier(transformer->weights_gpu.token_embedding_table);
    AddSRVBarrier(transformer->weights_gpu.rms_att_weight);
    AddSRVBarrier(transformer->weights_gpu.rms_ffn_weight);
    AddSRVBarrier(transformer->weights_gpu.wq);
    AddSRVBarrier(transformer->weights_gpu.wk);
    AddSRVBarrier(transformer->weights_gpu.wv);
    AddSRVBarrier(transformer->weights_gpu.wo);
    AddSRVBarrier(transformer->weights_gpu.w1);
    AddSRVBarrier(transformer->weights_gpu.w2);
    AddSRVBarrier(transformer->weights_gpu.w3);
    AddSRVBarrier(transformer->weights_gpu.rms_final_weight);
    if (!(transformer->config.vocab_size > 0 ? 1 : 0)) {
        AddSRVBarrier(transformer->weights_gpu.wcls);
    }

    commandList->ResourceBarrier(uavBarriers.size(), uavBarriers.data());

    commandList->Close();
    
    // Execute copy commands
    ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    
    // Wait for copy to complete
    WaitForGpu();
}

// Execute compute shader 
void ExecuteCompute(Transformer* transformer) {
    commandList->Reset(commandAllocator.Get(), pipelineState.Get());
    
    // Set descriptor heaps
    ID3D12DescriptorHeap* heaps[] = { computeHeap.Get() };
    commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    
    // Set compute pipeline
    commandList->SetComputeRootSignature(rootSignature.Get());
    commandList->SetComputeRootDescriptorTable(0, computeHeap->GetGPUDescriptorHandleForHeapStart());

    // Dispatch compute shader
    commandList->Dispatch(int(transformer->runstate_gpu.logits_size / 32), 1, 1);

    // Add barrier to ensure computation is complete
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        transformer->runstate_gpu.logits.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );
    commandList->ResourceBarrier(1, &barrier);

    // Copy result to read back buffer
    commandList->CopyResource(transformer->runstate_gpu.logits_readback.Get(), transformer->runstate_gpu.logits.Get());

    commandList->Close();

    // Execute commands
    ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Wait for computation to complete
    WaitForGpu();
}
// Read back computation result
std::vector<float> ReadBackResult(Transformer* transformer) {
    std::vector<float> result(transformer->runstate_gpu.logits_size);
    void* mappedData;
    transformer->runstate_gpu.logits_readback->Map(0, nullptr, &mappedData);
    memcpy(result.data(), mappedData, transformer->runstate_gpu.logits_size * sizeof(float));
    transformer->runstate_gpu.logits_readback->Unmap(0, nullptr);
    return result;
}


void memory_map_weights(TransformerWeights* w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;

    w->token_embedding_table = ptr;
    w->token_embedding_table_size = p->vocab_size * p->dim;
    ptr += w->token_embedding_table_size;

    w->rms_att_weight = ptr;
    w->rms_att_weight_size = n_layers * p->dim;
    ptr += w->rms_att_weight_size;

    w->wq = ptr;
    w->wq_size = n_layers * p->dim * (p->n_heads * head_size);
    ptr += w->wq_size;

    w->wk = ptr;
    w->wk_size = n_layers * p->dim * (p->n_kv_heads * head_size);
    ptr += w->wk_size;

    w->wv = ptr;
    w->wv_size = n_layers * p->dim * (p->n_kv_heads * head_size);
    ptr += w->wv_size;

    w->wo = ptr;
    w->wo_size = n_layers * (p->n_heads * head_size) * p->dim;
    ptr += w->wo_size;

    w->rms_ffn_weight = ptr;
    w->rms_ffn_weight_size = n_layers * p->dim;
    ptr += w->rms_ffn_weight_size;

    w->w1 = ptr;
    w->w1_size = n_layers * p->dim * p->hidden_dim;
    ptr += w->w1_size;

    w->w2 = ptr;
    w->w2_size = n_layers * p->hidden_dim * p->dim;
    ptr += w->w2_size;

    w->w3 = ptr;
    w->w3_size = n_layers * p->dim * p->hidden_dim;
    ptr += w->w3_size;

    w->rms_final_weight = ptr;
    w->rms_final_weight_size = p->dim;
    ptr += w->rms_final_weight_size;

    ptr += p->seq_len * head_size; // / skip what used to be freq_cis_real (for RoPE)

    w->wcls = shared_weights ? w->token_embedding_table : ptr;
    w->wcls_size = shared_weights ? 0 : p->vocab_size * p->dim;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
    HANDLE* file_handle, float** data, size_t* file_size) {
    FILE* file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

    // Read configuration information
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);

    // Get file size
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);

    *file_handle = CreateFileA(checkpoint, GENERIC_READ, FILE_SHARE_READ, NULL,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (*file_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "CreateFile failed!\n");
        exit(EXIT_FAILURE);
    }

    HANDLE mapping = CreateFileMapping(*file_handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mapping == NULL) {
        CloseHandle(*file_handle);
        fprintf(stderr, "CreateFileMapping failed!\n");
        exit(EXIT_FAILURE);
    }

    *data = (float*)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping); // Can close mapping handle after creating the view

    if (*data == NULL) {
        CloseHandle(*file_handle);
        fprintf(stderr, "MapViewOfFile failed!\n");
        exit(EXIT_FAILURE);
    }

    // Set weight pointers
    float* weights_ptr = *data + sizeof(Config) / sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void free_transformer(Transformer* t) {
    if (t->data != NULL) {
        UnmapViewOfFile(t->data);
    }
    if (t->file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(t->file_handle);
    }
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}


// Main function example
int main(int argc, char* argv[]) {
    // Initialize
        // default parameters
    char* checkpoint_path = "../model/stories110M.bin";  // e.g. out/model.bin
    char* tokenizer_path = "../model/tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char* prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char* mode = "generate";    // generate|chat
    char* system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    //if (argc >= 2) { checkpoint_path = argv[1]; }
    //else { error_usage(); }
    //for (int i = 2; i < argc; i += 2) {
    //    // do some basic validation
    //    if (i + 1 >= argc) { error_usage(); } // must have arg after flag
    //    if (argv[i][0] != '-') { error_usage(); } // must start with dash
    //    if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
    //    // read in the args
    //    if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
    //    else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
    //    else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
    //    else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
    //    else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
    //    else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
    //    else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
    //    else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
    //    else { error_usage(); }
    //}
    Transformer transformer;
    read_checkpoint(checkpoint_path, &transformer.config, &transformer.weights, &transformer.file_handle, &transformer.data, &transformer.file_size);

    InitializeDevice();
    CreateComputePipeline();
    CreateResources(&transformer);
    CreateSyncObjects();
    CreateUploadAndReadBackBuffers(&transformer);


    // Execute computation
    UploadTransformerWeights(&transformer);
    ExecuteCompute(&transformer);
    std::vector<float> result = ReadBackResult(&transformer);
    // Clean up resources
    CloseHandle(fenceEvent);
    return 0;
}
