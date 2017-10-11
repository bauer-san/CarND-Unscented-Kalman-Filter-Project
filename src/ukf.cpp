#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

#define TEST 0

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
  
  // calculate scaling factor
  lambda_ = 3 - n_aug_;

  // initial state vector
  x_ = VectorXd(n_x_);
//  x_.fill(0.);
 
  //matrix to hold the sigma points
  Xsig_aug = MatrixXd(n_aug_, n_sig_);

  //matrix to hold the "k+1" predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  
  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
//  P_.fill(0.);

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

  //Calculate the weights
  weights_ = VectorXd(n_sig_);
  for (int i=0;i<n_sig_;i++) {
    if (i==0) {
        weights_(i) = lambda_ / (lambda_+n_aug_);
    } else {
        weights_(i) = 0.5 / (lambda_+n_aug_);
    }
  }
//std::cout << weights_ << std::endl;   

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  if (!is_initialized_) {// initialize to the first measurement
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
  } else {
    
    Prediction((meas_package.timestamp_ - time_us_)/1E6);
    
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      UpdateLidar(meas_package);
    }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      UpdateRadar(meas_package);
    }
  }
  
  // Update time
  time_us_ = meas_package.timestamp_;
  
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

  //std::cout << "This delta_t: " << delta_t << std::endl;
  GenerateSigmaPoints();
  PredictSigmaPoints(delta_t);
  PredictMeanAndCovariance();  //MANUALLY unit tested to here
   
}

void UKF::GenerateSigmaPoints() {
  // *************************************************************************************
  // AUGMENT STATE VECTOR
  // *************************************************************************************
  // Create the augmented state vector be adding the means of the process noise and 
  // create the augmented covariance matrix to include the process noise covariance
  // *************************************************************************************

#if (TEST==1)
  //set example state
  x_ <<   5.7441,
         1.3800,
         2.2049,
         0.5015,
         0.3528;

  //create example covariance matrix
  P_ <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
          -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
           0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
          -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
          -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;
#endif

  //create augmented mean vector
  VectorXd x_aug_ = VectorXd(n_aug_);
  x_aug_.fill(0.0);
  x_aug_ << x_, // x_k|k
            0., // nu_a = [0., std_a_]
            0.; // nu_psi_dd = [0., std_yawdd_]

#if (TEST==1)
  //std::cout << "Prediction() x_\n" << x_ << std::endl;
  //std::cout << "x_aug_:\n" << x_aug_ << std::endl;  
  //std::cin >> n_aug_;  
#endif

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
  
  //calculate square root of P
  MatrixXd Psqrt = P_aug.llt().matrixL();
  
  MatrixXd temp = MatrixXd(n_aug_, n_aug_);
  temp = sqrt(lambda_+n_aug_)*Psqrt;

  Xsig_aug.col(0) << x_aug_;
  for (int col = 0; col<n_aug_; col++){
    Xsig_aug.col(col+1)      << x_aug_+temp.col(col);
    Xsig_aug.col(col+1+n_aug_) << x_aug_-temp.col(col);
  }

#if (TEST==1)  
  //std::cout << delta_t << std::endl;
  //std::cout << Xsig_aug << std::endl;
  //std::cout << P_aug << std::endl;
  //std::cin >> lambda_;
#endif  
//  expected result:
//Xsig_aug =
//5.7441 5.85768 5.7441 5.7441 5.7441 5.7441 5.7441 5.7441 5.63052 5.7441 5.7441 5.7441 5.7441 5.7441 5.7441
//1.38 1.34566 1.52806 1.38 1.38 1.38 1.38 1.38 1.41434 1.23194 1.38 1.38 1.38 1.38 1.38
//2.2049 2.28414 2.24557 2.29582 2.2049 2.2049 2.2049 2.2049 2.12566 2.16423 2.11398 2.2049 2.2049 2.2049 2.2049
//0.5015 0.44339 0.631886 0.516923 0.595227 0.5015 0.5015 0.5015 0.55961 0.371114 0.486077 0.407773 0.5015 0.5015 0.5015
//0.3528 0.299973 0.462123 0.376339 0.48417 0.418721 0.3528 0.3528 0.405627 0.243477 0.329261 0.22143 0.286879 0.3528 0.3528
//0 0 0 0 0 0 0.34641 0 0 0 0 0 0 -0.34641 0 
//0 0 0 0 0 0 0 0.34641 0 0 0 0 0 0 -0.34641

}

void UKF::PredictSigmaPoints(double delta_t) {
  // *************************************************************************************  
  // SIGMA POINT PREDICTION
  // *************************************************************************************
  // Plug in the sigma points to the process model to calculate Xsig_pred
  // *************************************************************************************  
#if (TEST==1)
     Xsig_aug <<
    5.7441,  5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;
		 
    delta_t = 0.1; //seconds
#endif
  double px_, py_, v_, psi_, psid_, nu_a_, nu_psidd_;
  
  //std::cout << "delta_t: " << delta_t;
  
  //predict sigma points
  for (int sigpt=0; sigpt<n_sig_; sigpt++){
    px_ =       Xsig_aug(0,sigpt);
    py_ =       Xsig_aug(1,sigpt);
    v_ =        Xsig_aug(2,sigpt);
    psi_ =      Xsig_aug(3,sigpt);
    psid_ =     Xsig_aug(4,sigpt);
    nu_a_ =     Xsig_aug(5,sigpt);
    nu_psidd_ = Xsig_aug(6,sigpt);
	
	//std::cout << "Xsig_aug:\n";
	//std::cout << px_ << std::endl;
	//std::cout << py_ << std::endl;
	//std::cout << v_ << std::endl;
	//std::cout << psi_ << std::endl;
	//std::cout << psid_ << std::endl;
	//std::cout << nu_a_ << std::endl;
	//std::cout << nu_psidd_ << std::endl;
    
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
#if (TEST==1)  
  //std::cout << "Xsig_pred_: \n" << Xsig_pred_ << std::endl;
  //std::cin >> px_;
#endif
//Xsig_pred =
//5.93553 6.06251 5.92217 5.9415 5.92361 5.93516 5.93705 5.93553 5.80832 5.94481 5.92935 5.94553 5.93589 5.93401 5.93553
//1.48939 1.44673 1.66484 1.49719 1.508 1.49001 1.49022 1.48939 1.5308 1.31287 1.48182 1.46967 1.48876 1.48855 1.48939
//2.2049 2.28414 2.24557 2.29582 2.2049 2.2049 2.23954 2.2049 2.12566 2.16423 2.11398 2.2049 2.2049 2.17026 2.2049
//0.53678 0.473387 0.678098 0.554557 0.643644 0.543372 0.53678 0.538512 0.600173 0.395462 0.519003 0.429916 0.530188 0.53678 0.535048
//0.3528 0.299973 0.462123 0.376339 0.48417 0.418721 0.3528 0.387441 0.405627 0.243477 0.329261 0.22143 0.286879 0.3528 0.318159  
}

void UKF::PredictMeanAndCovariance() {
  // *************************************************************************************  
  // PREDICT MEAN AND COVARIANCE
  // *************************************************************************************
  // Calculate the mean and covariance of Xsig_pred
  // *************************************************************************************  
#if (TEST==1)
        Xsig_pred_ <<
         5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;
		x_.fill(0.0);
		P_.fill(0.0);
#endif    
  
  //predict state mean
    for (int i=0;i<n_sig_; i++) {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

  VectorXd temp2 = VectorXd(n_x_);

    for (int i=0; i<n_sig_; i++) {
        temp2 = Xsig_pred_.col(i) - x_;
        P_ += weights_(i) * temp2 * temp2.transpose();
    }
#if (TEST==1)
  //std::cout << "x_:\n" << x_ << std::endl;
  //std::cout << "P_:\n" << P_ << std::endl;  
  //std::cin >> n_sig_;
#endif

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

  // LASER
  //update the state by using linear Kalman Filter equations
  //measurement matrix - laser

  VectorXd z = VectorXd(2);
  z = meas_package.raw_measurements_;

  MatrixXd R_ = MatrixXd(2,2);  
  R_.diagonal() << std_laspx_*std_laspx_, std_laspy_*std_laspy_;

  MatrixXd H_ = MatrixXd(2, 5);
  H_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0;

  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;  
  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;    

//Calculate NIS

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
