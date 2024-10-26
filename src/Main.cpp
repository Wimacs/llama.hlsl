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

void malloc_run_state(RunState* s, Config* p) {
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

void free_run_state(RunState* s) {
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

void memory_map_weights(TransformerWeights* w, Config* p, float* ptr, int shared_weights) {
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

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
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

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    Transformer t;
    read_checkpoint("./../model/stories110M.bin", &t.config, &t.weights, &t.fd, &t.data, &t.file_size);

    D3D12nBodyGravity sample(1280, 720, L"llama inference in hlsl on gpu");
    return Win32Application::Run(&sample, hInstance, nCmdShow);
}
