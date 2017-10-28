#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "ukf.h"
#include "ground_truth_package.h"
#include "measurement_package.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

int main() {
	
	UKF ukf;

	/*************************/
	/* GenerateSigmaPoints() */
	/*************************/
	std::cout << "Testing GenerateSigmaPoints()\n";
	ukf.std_a_ = 0.2;
	ukf.std_yawdd_ = 0.2;
	ukf.x_ << 5.7441,
			  1.3800,
			  2.2049,
		      0.5015,
		      0.3528;
	ukf.P_ << 0.0043, -0.0013, 0.0030, -0.0022, -0.0020,
		     -0.0013, 0.0077, 0.0011, 0.0071, 0.0060,
		      0.0030, 0.0011, 0.0054, 0.0007, 0.0008,
		     -0.0022, 0.0071, 0.0007, 0.0098, 0.0100,
		     -0.0020, 0.0060, 0.0008, 0.0100, 0.0123;
	ukf.GenerateSigmaPoints();
	std::cout << "Xsig_aug:\n" << ukf.Xsig_aug << std::endl;
	std::cout << "\nExpected Xsig_aug:\n";
	std::cout << "5.7441 5.85768 5.7441 5.7441 5.7441 5.7441 5.7441 5.7441 5.63052 5.7441 5.7441 5.7441 5.7441 5.7441 5.7441\n";
	std::cout << "1.38 1.34566 1.52806 1.38 1.38 1.38 1.38 1.38 1.41434 1.23194 1.38 1.38 1.38 1.38 1.38\n";
	std::cout << "2.2049 2.28414 2.24557 2.29582 2.2049 2.2049 2.2049 2.2049 2.12566 2.16423 2.11398 2.2049 2.2049 2.2049 2.2049\n";
	std::cout << "0.5015 0.44339 0.631886 0.516923 0.595227 0.5015 0.5015 0.5015 0.55961 0.371114 0.486077 0.407773 0.5015 0.5015 0.5015\n";
	std::cout << "0.3528 0.299973 0.462123 0.376339 0.48417 0.418721 0.3528 0.3528 0.405627 0.243477 0.329261 0.22143 0.286879 0.3528 0.3528\n";
	std::cout << "0 0 0 0 0 0 0.34641 0 0 0 0 0 0 -0.34641 0\n";
	std::cout << "0 0 0 0 0 0 0 0.34641 0 0 0 0 0 0 -0.34641\n";

	/************************/
	/* PredictSigmaPoints() */
	/************************/
	std::cout << "\n\nTesting PredictSigmaPoints()\n";
	ukf.Xsig_aug << 5.7441, 5.85768, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.63052, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441, 5.7441,
		1.38, 1.34566, 1.52806, 1.38, 1.38, 1.38, 1.38, 1.38, 1.41434, 1.23194, 1.38, 1.38, 1.38, 1.38, 1.38,
		2.2049, 2.28414, 2.24557, 2.29582, 2.2049, 2.2049, 2.2049, 2.2049, 2.12566, 2.16423, 2.11398, 2.2049, 2.2049, 2.2049, 2.2049,
		0.5015, 0.44339, 0.631886, 0.516923, 0.595227, 0.5015, 0.5015, 0.5015, 0.55961, 0.371114, 0.486077, 0.407773, 0.5015, 0.5015, 0.5015,
		0.3528, 0.299973, 0.462123, 0.376339, 0.48417, 0.418721, 0.3528, 0.3528, 0.405627, 0.243477, 0.329261, 0.22143, 0.286879, 0.3528, 0.3528,
		0, 0, 0, 0, 0, 0, 0.34641, 0, 0, 0, 0, 0, 0, -0.34641, 0,
		0, 0, 0, 0, 0, 0, 0, 0.34641, 0, 0, 0, 0, 0, 0, -0.34641;

	std::cout << "\n\nTesting PredictSigmaPoints()\n";

	ukf.PredictSigmaPoints(0.100);
	std::cout << "Xsig_pred_:\n" << ukf.Xsig_pred_ << std::endl;
	std::cout << "\nExpected Xsig_pred_:\n";
	std::cout << "5.93553 6.06251 5.92217 5.9415 5.92361 5.93516 5.93705 5.93553 5.80832 5.94481 5.92935 5.94553 5.93589 5.93401 5.93553\n";
	std::cout << "1.48939 1.44673 1.66484 1.49719 1.508 1.49001 1.49022 1.48939 1.5308 1.31287 1.48182 1.46967 1.48876 1.48855 1.48939\n";
	std::cout << "2.2049 2.28414 2.24557 2.29582 2.2049 2.2049 2.23954 2.2049 2.12566 2.16423 2.11398 2.2049 2.2049 2.17026 2.2049\n";
	std::cout << "0.53678 0.473387 0.678098 0.554557 0.643644 0.543372 0.53678 0.538512 0.600173 0.395462 0.519003 0.429916 0.530188 0.53678 0.535048\n";
	std::cout << "0.3528 0.299973 0.462123 0.376339 0.48417 0.418721 0.3528 0.387441 0.405627 0.243477 0.329261 0.22143 0.286879 0.3528 0.318159\n";

	/******************************/
	/* PredictMeanAndCovariance() */
	/******************************/
	std::cout << "\n\nTesting PredictMeanAndCovariance()\n";
	ukf.Xsig_pred_ <<
		5.9374, 6.0640, 5.925, 5.9436, 5.9266, 5.9374, 5.9389, 5.9374, 5.8106, 5.9457, 5.9310, 5.9465, 5.9374, 5.9359, 5.93744,
		1.48, 1.4436, 1.660, 1.4934, 1.5036, 1.48, 1.4868, 1.48, 1.5271, 1.3104, 1.4787, 1.4674, 1.48, 1.4851, 1.486,
		2.204, 2.2841, 2.2455, 2.2958, 2.204, 2.204, 2.2395, 2.204, 2.1256, 2.1642, 2.1139, 2.204, 2.204, 2.1702, 2.2049,
		0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337, 0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188, 0.5367, 0.535048,
		0.352, 0.29997, 0.46212, 0.37633, 0.4841, 0.41872, 0.352, 0.38744, 0.40562, 0.24347, 0.32926, 0.2214, 0.28687, 0.352, 0.318159;
	ukf.x_.fill(0.0);
	ukf.P_.fill(0.0);
	ukf.PredictMeanAndCovariance();
	std::cout << "x_:\n" << ukf.x_ << std::endl;
	std::cout << "\nExpected x_:\n";
	std::cout << "5.93637\n";
	std::cout << "1.49035\n";
	std::cout << "2.20528\n";
	std::cout << "0.536853\n";
	std::cout << "0.353577\n";

	std::cout << "P_:\n" << ukf.P_ << std::endl;
	std::cout << "\nExpected P_:\n";
	std::cout << "0.00543425 - 0.0024053 0.00341576 - 0.00348196 - 0.00299378\n";
	std::cout << "- 0.0024053 0.010845 0.0014923 0.00980182 0.00791091\n";
	std::cout << "0.00341576 0.0014923 0.00580129 0.000778632 0.000792973\n";
	std::cout << "- 0.00348196 0.00980182 0.000778632 0.0119238 0.0112491\n";
	std::cout << "- 0.00299378 0.00791091 0.000792973 0.0112491 0.0126972\n";

	return 0;
}