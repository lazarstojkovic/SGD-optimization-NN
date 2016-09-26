#pragma once
#include <string>
#include <iostream>
#include<math.h>

class layer {
protected:
	double learning_rate;
	double decay1;
	double decay2;
	double stepsize;
	double eps;
	std::string opt_alg;
	layer* next_layer;
	int nn;
	int ni;
	bool bias;
	double* weights;
	double* potentials;
	double* aux_mat1;
	double* aux_mat2;
	double* output;
	double* inputs;
	double* deltas;
	virtual double activation_fn(double a) = 0;
	virtual double activation_fn_prime(double a) = 0;
	virtual double d_ai_xj(int i, int j) = 0;
	virtual double d_ai_wij(int j) = 0;
	void compute_potentials(double *input) {
		for (int i = 0; i < ni; i++)
			inputs[i] = input[i];

		for (int i = 0; i < nn; i++) {
			potentials[i] = (bias) ? 1.0 * weights[i * (ni + 1)] : 0.0;
			for (int j = 0; j < ni; j++) {
				potentials[i] += input[j] * weights[i * (ni + bias) + j + bias];
			}
		}
	}
	double cost_fn_prime(double x, double y) {
		return -2.0 * (y - x);
	}
	double rms(double x) {
		return sqrt(x + eps);
	}
public:
	layer(int nn, int ni, bool bias, double* weights, double* deltas, layer* next_layer, double* aux_mat1, double* aux_mat2, std::string opt_alg, double learning_rate = 0.9, double decay1 = 0.9, double decay2 = 0.999, double stepsize = 0.001, double eps = 1.0e-8) {
		this->opt_alg = opt_alg;
		this->learning_rate = learning_rate;
		this->decay1 = decay1;
		this->decay2 = decay2;
		this->stepsize = stepsize;
		this->eps = eps;
		this->nn = nn;
		this->ni = ni;
		this->bias = bias;
		this->weights = weights;
		this->deltas = deltas;
		this->next_layer = next_layer;
		this->inputs = new double[ni];
		this->output = new double[nn];
		this->aux_mat1 = aux_mat1;
		this->aux_mat2 = aux_mat2;
		potentials = new double[nn]();
	}

	virtual ~layer() {
		delete[] potentials;
		delete[] inputs;
		delete[] output;
	}

	double* compute_output(double *input) {
		compute_potentials(input);
		for (int i = 0; i < nn; i++)
			output[i] = activation_fn(potentials[i]);
		return output;
	}

	void compute_deltas(double y = 0.0) {
		if (!next_layer) {
			deltas[0] = cost_fn_prime(output[0], y)*this->activation_fn_prime(potentials[0]);
		}
		else {
			for (int i = 0; i < nn; i++)
				deltas[i] = deltas[nn] * next_layer->d_ai_xj(0, i + 1)*this->activation_fn_prime(potentials[i]);
		}
	}

	void update_weights() {
		double grad, gs, us, upd, t = 0, m, v, mk, vk;
		for (int i = 0; i < nn; i++)
			for (int j = 0; j < ni + bias; j++) {
				grad = (!j) ? ((bias) ? deltas[i] * 1 : 0.0) : deltas[i] * this->d_ai_wij(j - bias);
				if (opt_alg == "SGD") {
					weights[i*(ni + bias) + j] -= learning_rate * grad;
				}
				else if (opt_alg == "Momentum") {
					weights[i*(ni + bias) + j] -= ((aux_mat1[i*(ni + bias) + j] *= decay1) += learning_rate*grad);
				}
				else if (opt_alg == "Adagrad") {
					gs = (aux_mat1[i*(ni + bias) + j] += grad*grad);
					weights[i*(ni + bias) + j] -= (learning_rate / rms(gs)) * grad;
				}
				else if (opt_alg == "Adadelta") {
					gs = (aux_mat1[i*(ni + bias) + j] *= decay1) += (1.0 - decay1)*grad*grad;
					upd = (rms(us = aux_mat2[i*(ni + bias) + j]) / rms(gs)) * grad;
					aux_mat2[i*(ni + bias) + j] = us * decay1 + (1.0 - decay1)*upd*upd;
					weights[i*(ni + bias) + j] -= upd;
				}
				else if (opt_alg == "Adam") {
					t++;
					m = (aux_mat1[i*(ni + bias) + j] *= decay1) += (1.0 - decay1)*grad;
					v = (aux_mat2[i*(ni + bias) + j] *= decay2) += (1.0 - decay2)*grad*grad;
					mk = m / (1.0 - pow(decay1, t));
					vk = v / (1.0 - pow(decay2, t));
					weights[i*(ni + bias) + j] -= stepsize * (mk / (sqrt(vk) + eps));
				}
			}
	}

	void set_opt_alg(std::string s) {
		this->opt_alg = s;
	}

	void set_aux_mat2(double *t) {
		this->aux_mat2 = t;
	}
	void set_aux_mat1(double *t) {
		this->aux_mat1 = t;
	}
};