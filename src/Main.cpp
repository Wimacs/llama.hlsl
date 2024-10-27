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

#include <corecrt_io.h>
#include <imgui.h>

#include "../DXSample/stdafx.h"
#include "D3D12llm.h"
#include "win.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, message, wParam, lParam))
        return true;

    //switch (message)
    //{
    //case WM_DESTROY: 
    //case WM_KILLFOCUS:
    //case WM_INPUT:
    //}
    return DefWindowProc(hWnd, message, wParam, lParam);
}


_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{

    Transformer t;
    read_checkpoint("./../model/stories110M.bin", &t.config, &t.weights, &t.fd, &t.data, &t.file_size);

    D3D12llm llm(1280, 720, L"llama inference in hlsl on gpu", &t);
    return Win32Application::Run(&llm, hInstance, nCmdShow);
}
