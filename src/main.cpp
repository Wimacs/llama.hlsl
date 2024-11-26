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

// Constants definition
const int MATRIX_WIDTH = 8192;
const int MATRIX_HEIGHT = 8192;
const UINT MATRIX_ELEMENTS = MATRIX_WIDTH * MATRIX_HEIGHT;

// DX12 related variables
ComPtr<ID3D12Device> device;
ComPtr<ID3D12CommandQueue> commandQueue;
ComPtr<ID3D12CommandAllocator> commandAllocator;
ComPtr<ID3D12GraphicsCommandList> commandList;
ComPtr<ID3D12RootSignature> rootSignature;
ComPtr<ID3D12PipelineState> pipelineState;

// Resource related
ComPtr<ID3D12Resource> matrixABuffer;
ComPtr<ID3D12Resource> matrixBBuffer;
ComPtr<ID3D12Resource> resultBuffer;

// Synchronization related variables
ComPtr<ID3D12Fence> fence;
UINT64 fenceValue = 0;
HANDLE fenceEvent;

// Upload heap buffers
ComPtr<ID3D12Resource> uploadBufferA;
ComPtr<ID3D12Resource> uploadBufferB;
ComPtr<ID3D12Resource> readBackBuffer;

// Add in global variable area ...
ComPtr<ID3D12DescriptorHeap> computeHeap;
UINT descriptorSize;

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

ComPtr<IDXCoreAdapter1> GetAdapter() {
    ComPtr<IDXCoreAdapterFactory> factory;
    if (FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory)))) {
        throw std::runtime_error("Failed to create DXCore factory");
    }

    ComPtr<IDXCoreAdapterList> adapterList;
    if (FAILED(factory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, 
        IID_PPV_ARGS(&adapterList)))) {
        throw std::runtime_error("Failed to create adapter list");
    }

    std::vector<ComPtr<IDXCoreAdapter1>> dxCoreAdapters;
    uint32_t adapterCount = adapterList->GetAdapterCount();
    
    for (uint32_t i = 0; i < adapterCount; ++i) {
        ComPtr<IDXCoreAdapter1> adapter;
        if (SUCCEEDED(adapterList->GetAdapter(i, IID_PPV_ARGS(&adapter)))) {
            bool isHardware = false;
        	{
                char description[128];
                (SUCCEEDED(adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, description)));
                std::cout << i + 1 << ". " << description << std::endl;

                // 获取显存信息
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
    heapDesc.NumDescriptors = 3; // 3 UAVs
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&computeHeap));
    
    descriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    // Create root signature
    CD3DX12_ROOT_PARAMETER rootParameter;
    CD3DX12_DESCRIPTOR_RANGE ranges[1];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 3, 0); // 3 UAVs, starting register 0
    rootParameter.InitAsDescriptorTable(1, ranges);

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


// Create resource buffers
void CreateResources() {
    // Create matrix buffers
    auto heapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    auto resourceDesc = CD3DX12_RESOURCE_DESC::Buffer(
        MATRIX_ELEMENTS * sizeof(float),
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
    );

    device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&matrixABuffer)
    );

    device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&matrixBBuffer)
    );

    device->CreateCommittedResource(
        &heapProperties,
        D3D12_HEAP_FLAG_NONE,
        &resourceDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(&resultBuffer)
    );

    // Create UAV descriptors
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = MATRIX_ELEMENTS;
    uavDesc.Buffer.StructureByteStride = sizeof(float);
    uavDesc.Buffer.CounterOffsetInBytes = 0;
    uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(computeHeap->GetCPUDescriptorHandleForHeapStart());
    
    device->CreateUnorderedAccessView(matrixABuffer.Get(), nullptr, &uavDesc, handle);
    
    handle.Offset(descriptorSize);
    device->CreateUnorderedAccessView(matrixBBuffer.Get(), nullptr, &uavDesc, handle);
    
    handle.Offset(descriptorSize);
    device->CreateUnorderedAccessView(resultBuffer.Get(), nullptr, &uavDesc, handle);


    
}

// Create synchronization objects
void CreateSyncObjects() {
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
}

// Create upload and read back buffers
void CreateUploadAndReadBackBuffers() {
    auto uploadHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
    auto readbackHeapProperties = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(MATRIX_ELEMENTS * sizeof(float));

    // Create upload buffer
    device->CreateCommittedResource(
        &uploadHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&uploadBufferA)
    );

    device->CreateCommittedResource(
        &uploadHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&uploadBufferB)
    );

    // Create read back buffer
    device->CreateCommittedResource(
        &readbackHeapProperties,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readBackBuffer)
    );
}

// Upload matrix data to GPU
void UploadMatrixData(const std::vector<float>& matrixA, const std::vector<float>& matrixB) {
    // Map and copy data to upload buffers
    void* mappedData;
    uploadBufferA->Map(0, nullptr, &mappedData);
    memcpy(mappedData, matrixA.data(), MATRIX_ELEMENTS * sizeof(float));
    uploadBufferA->Unmap(0, nullptr);

    uploadBufferB->Map(0, nullptr, &mappedData);
    memcpy(mappedData, matrixB.data(), MATRIX_ELEMENTS * sizeof(float));
    uploadBufferB->Unmap(0, nullptr);

    // Record copy commands
    commandList->Reset(commandAllocator.Get(), nullptr);

    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        matrixABuffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_DEST
    );
    commandList->ResourceBarrier(1, &barrier);

    barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        matrixBBuffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_DEST
    );
    commandList->ResourceBarrier(1, &barrier);

    commandList->CopyResource(matrixABuffer.Get(), uploadBufferA.Get());
    commandList->CopyResource(matrixBBuffer.Get(), uploadBufferB.Get());

    barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        matrixABuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    commandList->ResourceBarrier(1, &barrier);

    barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        matrixBBuffer.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    commandList->ResourceBarrier(1, &barrier);

    commandList->Close();
    
    // Execute copy commands
    ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    
    // Wait for copy to complete
    WaitForGpu();
}

// Execute compute shader
void ExecuteCompute() {
    commandList->Reset(commandAllocator.Get(), pipelineState.Get());
    
    // Set descriptor heaps
    ID3D12DescriptorHeap* heaps[] = { computeHeap.Get() };
    commandList->SetDescriptorHeaps(_countof(heaps), heaps);
    
    // Set compute pipeline
    commandList->SetComputeRootSignature(rootSignature.Get());
    commandList->SetComputeRootDescriptorTable(0, computeHeap->GetGPUDescriptorHandleForHeapStart());

    // Dispatch compute shader
    commandList->Dispatch(MATRIX_WIDTH / 32, MATRIX_HEIGHT / 32, 1);

    // Add barrier to ensure computation is complete
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        resultBuffer.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );
    commandList->ResourceBarrier(1, &barrier);

    // Copy result to read back buffer
    commandList->CopyResource(readBackBuffer.Get(), resultBuffer.Get());

    commandList->Close();

    // Execute commands
    ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Wait for computation to complete
    WaitForGpu();
}

// Read back computation result
std::vector<float> ReadBackResult() {
    std::vector<float> result(MATRIX_ELEMENTS);
    void* mappedData;
    readBackBuffer->Map(0, nullptr, &mappedData);
    memcpy(result.data(), mappedData, MATRIX_ELEMENTS * sizeof(float));
    readBackBuffer->Unmap(0, nullptr);
    return result;
}



// Main function example
int main() {
    // Initialize
    InitializeDevice();
    CreateComputePipeline();
    CreateResources();
    CreateSyncObjects();
    CreateUploadAndReadBackBuffers();

    // Prepare test data
    std::vector<float> matrixA(MATRIX_ELEMENTS, 1.0f);  // Example data
    std::vector<float> matrixB(MATRIX_ELEMENTS, 2.0f);  // Example data

    // Execute computation
    UploadMatrixData(matrixA, matrixB);
    ExecuteCompute();
    std::vector<float> result = ReadBackResult();
    // Clean up resources
    CloseHandle(fenceEvent);
    return 0;
}
