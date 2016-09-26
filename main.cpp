#include "neural_network.h"
#include <time.h>
#include <string>
#include <fstream>

int main() {	
	std::string opt_alg = "Adam";
	std::string opt_alg2 = "Adam";
													//alg,   in, hid,   out, lr, dec1, dec2, steps, eps
	neural_network < tanh_layer, sig_layer > neural_net(opt_alg, 1, 20, 1, 0.1, 0.99, 0.999, 0.01, 1.0e-8);
	double *arr = new double[1];
	srand(time(NULL));
	int n = 10000;
	double errp=0.0;
	std::cout << opt_alg << std::endl;
	for (int i = 0; i < n; i++) {
		if (i == n / 4 && opt_alg2 != opt_alg) {
			neural_net.set_opt_alg(opt_alg2);
			std::cout << opt_alg2 << std::endl;
		}
		arr[0] = (double)rand() / (double)RAND_MAX * 4.0 * acos(0.0);
		double res = sin(arr[0]);
		std::cout << "trainPASS: " << i+1 << "; IN: " << arr[0]<< "; expOUT: " << res << "; netOUT: " <<
			2.0*neural_net.train_network_pass(arr, 0.5*res + 0.5)[0] - 1.0 << std::endl;
		std::cout << std::endl;
		errp += neural_net.pass_error(0.5*res + 0.5);
	}
	
	double t = errp / n;

	errp = 0.0;
	for (int i = 0; i < n/2; i++) {
		arr[0] = (double)rand() / (double)RAND_MAX* 4.0 * acos(0.0);
		double res = sin(arr[0]);
		std::cout << "testPASS: " << i + 1 << "; IN: " << arr[0] << "; expOUT: " << res << "; netOUT: " <<
			2.0*neural_net.compute_output(arr)[0]-1.0 << std::endl;
		std::cout << std::endl;
		errp += neural_net.pass_error(0.5*res + 0.5);
	}
	neural_net.printWeights();
	std::cout << std::endl << "trainAcc: " << t << "; testAcc: " << errp / n / 2 << std::endl << std::endl;
	delete[] arr;
}