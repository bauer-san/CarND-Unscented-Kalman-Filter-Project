#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // starts off not initialized
  is_initialized_ = false;

  // dimension of state vector  
  n_x_ = n_states;
  
  // dimension of augmented state vector  
  n_aug_ = n_x_ + 2;

  // calculate number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // initial state vector
  x_ = VectorXd(n_x_);
//  x_.fill(0.);
  
  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
//  P_.fill(0.);

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2; //30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 10*M_PI/180; //30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
   
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  if (is_initialized_) {
  } else { // initialize to the first measurement
    switch  (meas_package.sensor_type_) {
	  case MeasurementPackage::LASER:
        x_(px) = meas_package.raw_measurements_(0);
	    x_(py) = meas_package.raw_measurements_(1);
	    x_(v) = 0.;
	    x_(psi) = atan(meas_package.raw_measurements_(1)/meas_package.raw_measurements_(0));
	    x_(psid) = 0.;
	  
        P_.diagonal() << std_laspx_*std_laspx_, std_laspy_*std_laspy_, 1., 10*M_PI/180, 1.;
		
		is_initialized_ = true;
		break;
	  case MeasurementPackage::RADAR:
        //rho, psi, psid
        x_(px) = meas_package.raw_measurements_(0) * cos(meas_package.raw_measurements_(1));
	    x_(py) = meas_package.raw_measurements_(0) * sin(meas_package.raw_measurements_(1));
	    x_(v) = 0.;
	    x_(psi) = meas_package.raw_measurements_(1);
	    x_(psid) = 0.;	  
	  
	    P_.diagonal() << std_radr_*std_radr_, std_radr_*std_radr_, std_radrd_*std_radrd_, std_radphi_*std_radphi_, 1.;
		
		is_initialized_ = true;
		break;
	  default:
	    std::cout << "UNKNOWN SENSOR TYPE: " << meas_package.sensor_type_ << std::endl;
    }
  //std::cout << "x_ after initialization:\n" << x_ << std::endl;
  //std::cout << "P_ after initialization:\n" << P_ << std::endl;
    
  }
  
  //std::cout << "x_ before prediction:\n" << x_ << std::endl;
  // Prediction step
  Prediction((meas_package.timestamp_ - time_us_)/1E6);
  //std::cout << "x_ after prediction:\n" << x_ << std::endl;
  
  // Update time
  time_us_ = meas_package.timestamp_;
  
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // Prediction includes the following steps:
  // 1. Generate Sigma Points
  // 2. Predict Sigma Points
  // 3. Predict Mean and Covariance

  //std::cout << delta_t << std::endl;
  
  // *************************************************************************************
  // AUGMENT STATE VECTOR
  // *************************************************************************************
  // Create the augmented state vector be adding the means of the process noise and 
  // create the augmented covariance matrix to include the process noise covariance
  // *************************************************************************************

  //create augmented mean vector
  VectorXd x_aug_ = VectorXd(n_aug_);

  //create augmented mean state
  x_aug_ << x_,
            0., // nu_a = [0., std_a_]
            0.; // nu_psi_dd = [0., std_yawdd_]

  //std::cout << "Prediction() x_\n" << x_ << std::endl;
  //std::cout << "x_aug_\n" << x_aug_ << std::endl;  

  //create augmented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) << P_;
  P_aug.bottomRightCorner(n_aug_-n_x_, n_aug_-n_x_) << std_a_*std_a_, 
                                                                  0., 
                                                                  0., 
                                               std_yawdd_*std_yawdd_;

  // *************************************************************************************
  // GENERATE SIGMA POINTS
  // *************************************************************************************
  // Use sigma points to represent the uncertainty of the posterior state estimation
  // Xsig = [ x_  x_+sqrt((lambda_ +n_x_) * P)  x_-sqrt((lambda_ +n_x_) * P)]
  // *************************************************************************************

  //create a matrix to hold the sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  
  //calculate square root of P
  MatrixXd Psqrt = P_aug.llt().matrixL();
  
  // calculate scaling factor
  lambda_ = 3. - n_x_;
  
  MatrixXd temp = MatrixXd(n_aug_, n_aug_);
  temp = sqrt(lambda_+n_aug_)*Psqrt;

  Xsig_aug.col(0) << x_aug_;
  for (int col = 0; col<n_aug_; col++){
    Xsig_aug.col(col+1)      << x_aug_+temp.col(col);
    Xsig_aug.col(col+1+n_x_) << x_aug_-temp.col(col);
  }
  
  //std::cout << delta_t << std::endl;
  //std::cout << Xsig_aug << std::endl;
  //std::cout << P_aug << std::endl;

  // *************************************************************************************  
  // SIGMA POINT PREDICTION
  // *************************************************************************************
  // Plug in the sigma points to the process model to calculate Xsig_pred
  // *************************************************************************************  
  float px_,py_,v_,psi_,psid_,nu_a_,nu_psidd_;
  
  //predict sigma points
  for (int sigpt=0; sigpt<n_sig_; sigpt++){
    px_ =       Xsig_aug(0,sigpt);
    py_ =       Xsig_aug(1,sigpt);
    v_ =        Xsig_aug(2,sigpt);
    psi_ =      Xsig_aug(3,sigpt);
    psid_ =     Xsig_aug(4,sigpt);
    nu_a_ =     Xsig_aug(5,sigpt);
    nu_psidd_ = Xsig_aug(6,sigpt);
    
   if (fabs(psid_)< 0.001) { //avoid division by zero
     Xsig_pred_(0,sigpt) = px_   + v_*cos(psi_)*delta_t + 0.5*delta_t*delta_t*cos(psi_)*nu_a_;
     Xsig_pred_(1,sigpt) = py_   + v_*sin(psi_)*delta_t + 0.5*delta_t*delta_t*sin(psi_)*nu_a_;
     Xsig_pred_(2,sigpt) = v_    + 0                    + delta_t*nu_a_;
     Xsig_pred_(3,sigpt) = psi_  + psid_*delta_t        + 0.5*delta_t*delta_t*nu_psidd_;
     Xsig_pred_(4,sigpt) = psid_ + 0                    + delta_t*nu_psidd_;
    } else {
     Xsig_pred_(0,sigpt) = px_   + v_/psid_*(sin(psi_+psid_*delta_t)-sin(psi_))  + 0.5*delta_t*delta_t*cos(psi_)*nu_a_;
     Xsig_pred_(1,sigpt) = py_   + v_/psid_*(-cos(psi_+psid_*delta_t)+cos(psi_)) + 0.5*delta_t*delta_t*sin(psi_)*nu_a_;
     Xsig_pred_(2,sigpt) = v_    + 0                                             + delta_t*nu_a_;
     Xsig_pred_(3,sigpt) = psi_  + psid_*delta_t                                 + 0.5*delta_t*delta_t*nu_psidd_;
     Xsig_pred_(4,sigpt) = psid_ + 0                                             + delta_t*nu_psidd_;
    }
  }
  //std::cout << "Xsig_pred_: \n" << Xsig_pred_ << std::endl;
  
  // *************************************************************************************  
  // PREDICT MEAN AND COVARIANCE
  // *************************************************************************************
  // Calculate the mean and covariance of Xsig_pred
  // *************************************************************************************  
  
  //Calculate the weights
  weights_ = VectorXd(n_sig_);
  lambda_ = 3  - n_aug_;  // NOTE: this value for lambda_ is calculated with n_aug_, rather than n_x_
  for (int i=0;i<n_sig_;i++) {
    if (i==0) {
        weights_(i) = lambda_ / (lambda_+n_aug_);
    } else {
        weights_(i) = 0.5 / (lambda_+n_aug_);
    }
  }

//std::cout << weights_ << std::endl;
//std::cout << x_ << std::endl;
  
  //predict state mean
    for (int i=0;i<n_sig_; i++) {
std::cout << "calculation:\n" << weights_(i) * Xsig_pred_.col(i) << std::endl;
std::cin >> i;
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

  VectorXd temp2 = VectorXd(n_x_);

    for (int i=0; i<n_sig_; i++) {
        temp2 = Xsig_pred_.col(i) - x_;
        P_ += weights_(i) * temp2 * temp2.transpose();
    }

  std::cout << x_ << std::endl;
 
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
