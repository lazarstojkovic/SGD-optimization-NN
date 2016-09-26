#pragma once

#include "layer.h"

class linear_layer : public layer {
protected:
	virtual double activation_fn(double a) {
		return a;
	}
	virtual double activation_fn_prime(double a) {
		return 1.0;
	}
	double d_ai_xj(int i, int j) {
		return weights[i * (ni + bias) + j];
	}
	double d_ai_wij(int j) {
		return inputs[j];
	}
public:
	linear_layer(int nn, int ni, int bias, double* weights, double* deltas, layer* next_layer, double* aux_mat1, double* aux_mat2, std::string opt_alg, double learning_rate = 0.9, double decay1 = 0.9, double decay2 = 0.999, double stepsize = 0.001, double eps = 1.0e-8) :
		layer(nn, ni, bias, weights, deltas, next_layer, aux_mat1, aux_mat2, opt_alg, learning_rate, decay1, decay2, stepsize, eps) { }
};