#pragma once

//normalizes the layers of the transformer to improve training stability and convergenc(kernel).

void layernorm_forward(
    float* output, //normalized values.
    const float* input,
    const float* gamma,//scaling parameter(learned). initialised to 1.0.
    const float* beta,// learned shift parameter. initialised to 0.0
    int N,// total token bieng normalised.
    int d_model,
    float eps = 1e-5f//small constant added to variance before taking the sqrt. prevents /0.
);

void layernorm_backward(
    float* d_input, //gradient w.r.t input.
    float* d_gamma, //gradient w.r.t gamma.
    float* d_beta, //gradient w.r.t beta.
    const float* d_output, //gradient w.r.t output.
    const float* input,
    const float* gamma,
    int N,
    int d_model,
    float eps = 1e-5f
); // d_gamma and d_beta are accumulated(+=) not set, cuz they recieve gradient contri from all tokens, and tokens are processes in parallel.