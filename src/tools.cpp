#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
	VectorXd rmse(4);
	rmse.fill(0.0);

	// check the validity of the inputs:
	if (estimations.size()==0) {        //  * the estimation vector size should not be zero
	    std::cout << "CalculateRMSE() - ERROR - zero length estimation vector";
	}
	if (estimations.size() != ground_truth.size()) {//  * the estimation vector size should equal ground truth vector size
	    std::cout << "CalculateRMSE() - ERROR - estimation and ground_truth vectors are different size";
	}

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
	    VectorXd residual = estimations[i]-ground_truth[i];
        residual = residual.array()*residual.array();
        rmse += residual;
	}

	//calculate the mean
    rmse = rmse / estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	return rmse;  
}
