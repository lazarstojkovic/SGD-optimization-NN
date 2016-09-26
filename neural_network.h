#pragma once
#include<iostream>
#include <random>
#include <string>
#include "linear_layer.h"
#include "tanh_layer.h"
#include "sig_layer.h"

const double min_ran = -0.1;
const double max_ran = 0.1;

template<typename H, typename O>
class neural_network {
protected:
	int n_layers;
	layer** layers;
	int in_num, hl_num, ol_num;
	int weights_size;
	double* weights;
	double* output;
	double* deltas;
	double* aux_mat1;
	double* aux_mat2;
	double *a1_shifted;
	double *a2_shifted;
	std::string opt_alg;
public:
	neural_network(std::string opt_alg, int in_num, int hl_num, int ol_num = 1, double learning_rate = 0.9, double decay1 = 0.9, double decay2 = 0.999, double stepsize = 0.001, double eps = 1.0e-8) {
		n_layers = 2;
		this->opt_alg = opt_alg;
		this->in_num = in_num;
		this->hl_num = hl_num;
		this->ol_num = ol_num;
		this->weights_size = (in_num + 1) * hl_num + (hl_num + 1) * ol_num;
		aux_mat1 = nullptr;
		aux_mat2 = nullptr;
		output = nullptr;
		a1_shifted = nullptr;
		a2_shifted = nullptr;
		initialize_weights();
		initialize_deltas(hl_num + ol_num);
		set_aux_params(opt_alg);
		layers = new layer*[n_layers];
		layers[1] = new O(ol_num, hl_num, 1, weights + (in_num + 1) * hl_num, deltas + hl_num, nullptr, a1_shifted, a2_shifted, opt_alg, learning_rate, decay1, decay2, stepsize, eps);
		layers[0] = new H(hl_num, in_num, 1, weights, deltas, layers[1], aux_mat1, aux_mat2, opt_alg, learning_rate, decay1, decay2, stepsize, eps);
	}

	virtual ~neural_network() {
		delete layers[0];
		delete layers[1];
		delete[] layers;
		delete[] weights;
		delete[] deltas;
		if (opt_alg != "SGD") {
			delete[] aux_mat1;
			if (opt_alg != "Adagrad"  && opt_alg != "Momentum")
				delete[] aux_mat2;
		}
	}

	void printWeights() {
		for (int i = 0; i < hl_num; i++) {
			for (int j = 0; j < in_num + 1; j++)
				std::cout << weights[i*(in_num + 1) + j] << " ";
			std::cout << std::endl;
		}
		std::cout << std::endl;
		for (int i = (in_num + 1) * hl_num; i < (in_num + 1) * hl_num + (hl_num + 1) * ol_num; i++)
			std::cout << weights[i] << " ";
		std::cout << std::endl;
	}

	double* train_network_pass(double *input, double ex_res) {
		double* ret = compute_output(input);
		back_propagate(ex_res);
		return ret;
	}

	double pass_error(double x) {
		return abs(output[0] - x);
	}

	double* compute_output(double *input) {
		output = layers[1]->compute_output(layers[0]->compute_output(input));
		return output;
	}

	void set_opt_alg(std::string s) {
		this->opt_alg = s;
		set_aux_params(s);
		layers[0]->set_opt_alg(s);
		layers[1]->set_opt_alg(s);
		layers[0]->set_aux_mat1(aux_mat1);
		layers[1]->set_aux_mat1(aux_mat1 + (in_num + 1) * hl_num);
		layers[0]->set_aux_mat2(aux_mat2);
		layers[1]->set_aux_mat2(aux_mat2 + (in_num + 1) * hl_num);
	}

private:
	void initialize_weights() {
		weights = new double[weights_size];
		std::random_device device;
		std::default_random_engine engine(device());
		std::uniform_real_distribution<double> distribution(min_ran, max_ran);
		for (int i = 0; i < weights_size; i++)
			weights[i] = distribution(engine);
	}

	void initialize_deltas(int deltas_size) {
		deltas = new double[deltas_size]();
	}

	void set_aux_params(std::string s) {
		if (s != "SGD") {
			initialize_aux_mats();
			a1_shifted = aux_mat1 + (in_num + 1) * hl_num;
			if (s != "Adagrad"  && s != "Momentum")
				a2_shifted = aux_mat2 + (in_num + 1) * hl_num;
		}
	}
	void initialize_aux_mats() {
		//if (!aux_mat1)
			aux_mat1 = new double[weights_size]();
		if (opt_alg != "Adagrad" && opt_alg != "Momentum" )// && !aux_mat2) 
			aux_mat2 = new double[weights_size]();
	}

	void back_propagate(double p = 0.0) {
		layers[1]->compute_deltas(p);
		layers[0]->compute_deltas();
		layers[1]->update_weights();
		layers[0]->update_weights();
	}
};