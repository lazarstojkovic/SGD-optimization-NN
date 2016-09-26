#pragma once

#include "linear_layer.h"

class tanh_layer : public linear_layer {
private:
	double activation_fn(double a) {
		return (exp(2.0*a) - 1.0) / (exp(2.0*a) + 1.0);
	}
	double activation_fn_prime(double a) {
		return 1.0 - activation_fn(a) * activation_fn(a);
	}
public:
	tanh_layer(int nn, int ni, int bias, double* weights, double* deltas, layer* next_layer, double* aux_mat1, double* aux_mat2, std::string opt_alg, double learning_rate = 0.9, double decay1 = 0.9, double decay2 = 0.999, double stepsize = 0.001, double eps = 1.0e-8) :
		linear_layer(nn, ni, bias, weights, deltas, next_layer, aux_mat1, aux_mat2, opt_alg, learning_rate, decay1, decay2, stepsize, eps) {}
};