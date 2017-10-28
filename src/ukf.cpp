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
  
  ///* number of lidar measurements
  n_z_lidar_ = 2; // px, py

  ///* number of radar measurements
  n_z_radar_ = 3; //rho, psi, rhodot

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2; // 2; //30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2; // 10 * M_PI / 180; //30;

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

  NIS_laser_ = 0.;
  NIS_radar_ = 0.;

  // initial state vector
  x_ = VectorXd(n_x_);
  x_.fill(0.);
 
  //matrix to hold the sigma points
  Xsig_aug = MatrixXd(n_aug_, n_sig_);
  Xsig_aug.fill(0.);

  //matrix to hold the "k+1" predicted sigma points
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  Xsig_pred_.fill(0.);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_.fill(0.);

  // lidar measurement matrix
  H_lidar_ = MatrixXd(n_z_lidar_, n_x_);
  H_lidar_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  ///* lidar covariance matrix    
  R_lidar_ = MatrixXd(n_z_lidar_, n_z_lidar_);  
  R_lidar_.fill(0.);
  R_lidar_.diagonal() << std_laspx_*std_laspx_, std_laspy_*std_laspy_;			  

  //Calculate the weights
  weights_ = VectorXd(n_sig_);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_/(lambda_ + n_aug_);
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
//  std::cout << "x_ after initialization:\n" << x_ << std::endl;
//  std::cout << "P_ after initialization:\n" << P_ << std::endl;
  } else {
    
    Prediction((meas_package.timestamp_ - time_us_)/1E6);
    
    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
//	  std::cout << "laser\n";
      UpdateLidar(meas_package);
    }

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
//	  std::cout << "radar\n";
      UpdateRadar(meas_package);
    }
  }
  
  // Update time
  time_us_ = meas_package.timestamp_;

#if (TEST==1)
	//std::cout << "x_:\n" << x_ << std::endl;
	//std::cout << "P_:\n" << P_ << std::endl;
	//std::cin >> is_initialized_;
#endif
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
#if (TEST==1)
//	std::cout << "Xsig_aug:\n" << Xsig_aug << std::endl;
#endif

  PredictSigmaPoints(delta_t);
#if (TEST==1)
//	std::cout << "Xsig_pred_:\n" << Xsig_pred_ << std::endl;
#endif

  PredictMeanAndCovariance();
#if (TEST==1)
//  std::cout << "x_(k+1|k):\n" << x_ << std::endl;
//  std::cout << "P_(k+1|k):\n" << P_ << std::endl;
//  std::cin >> lambda_;
#endif   
}

void UKF::GenerateSigmaPoints() {
  // *************************************************************************************
  // AUGMENT STATE VECTOR
  // *************************************************************************************
  // Create the augmented state vector be adding the means of the process noise and 
  // create the augmented covariance matrix to include the process noise covariance
  // *************************************************************************************

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
  //std::cout << "Xsig_aug:\n" << Xsig_aug << std::endl;
  //std::cout << "P_aug:\n" << P_aug << std::endl;
  //std::cin >> lambda_;
#endif  


}

void UKF::PredictSigmaPoints(double delta_t) {
  // *************************************************************************************  
  // SIGMA POINT PREDICTION
  // *************************************************************************************
  // Plug in the sigma points to the process model to calculate Xsig_pred
  // *************************************************************************************  

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
      Xsig_pred_(2,sigpt) = v_    + 0.                   + delta_t*nu_a_;
      Xsig_pred_(3,sigpt) = psi_  + psid_*delta_t        + 0.5*delta_t*delta_t*nu_psidd_;
      Xsig_pred_(4,sigpt) = psid_ + 0.                   + delta_t*nu_psidd_;
    } else {
      Xsig_pred_(0,sigpt) = px_   + v_/psid_*(sin(psi_+psid_*delta_t)-sin(psi_))  + 0.5*delta_t*delta_t*cos(psi_)*nu_a_;
      Xsig_pred_(1,sigpt) = py_   + v_/psid_*(-cos(psi_+psid_*delta_t)+cos(psi_)) + 0.5*delta_t*delta_t*sin(psi_)*nu_a_;
      Xsig_pred_(2,sigpt) = v_    + 0.                                            + delta_t*nu_a_;
      Xsig_pred_(3,sigpt) = psi_  + psid_*delta_t                                 + 0.5*delta_t*delta_t*nu_psidd_;
      Xsig_pred_(4,sigpt) = psid_ + 0.                                            + delta_t*nu_psidd_;
    }
  }
#if (TEST==1)  
  //std::cout << "Xsig_pred_: \n" << Xsig_pred_ << std::endl;
  //std::cin >> px_;
#endif

}

void UKF::PredictMeanAndCovariance() {
  // *************************************************************************************  
  // PREDICT MEAN AND COVARIANCE
  // *************************************************************************************
  // Calculate the mean and covariance of Xsig_pred
  // *************************************************************************************  
  
  //predict new state mean
	x_.fill(0.);
	x_ = Xsig_pred_ * weights_;

  VectorXd temp2 = VectorXd(n_x_);
    P_.fill(0.);
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
  // LASER
  //update the state by using linear Kalman Filter equations
  //measurement matrix - laser

  VectorXd z = VectorXd(n_z_lidar_);
  z = meas_package.raw_measurements_;

  VectorXd z_pred = H_lidar_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_lidar_.transpose();
  MatrixXd PHt = P_ * Ht;  
  MatrixXd S = H_lidar_ * PHt + R_lidar_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_lidar_) * P_;    

  //Calculate Normalized Innovation Squared (NIS)
  NIS_laser_ = y.transpose() * Si * y;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_radar_, n_sig_);  

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_radar_);
  
  VectorXd z_diff = VectorXd(n_z_radar_);  

  float px_, py_, v_, psi_, psid_;
  //transform sigma points into measurement space
  for (int i=0;i<n_sig_;i++) {
      px_ = Xsig_pred_.col(i)(px);
      py_ = Xsig_pred_.col(i)(py);
      v_ = Xsig_pred_.col(i)(v);
      psi_ = Xsig_pred_.col(i)(psi);
      psid_ = Xsig_pred_.col(i)(psid);
      
	  // Prevent divide by zero
	  if (px_==0.) {
		px_ = 0.00001; //just a small x
	  }
	  if (py_==0.) {
		py_ = 0.00001; //just a small y
	  }
	  
      Zsig.col(i) <<                                       sqrt(px_*px_ + py_*py_),   // r
                                                                   atan2(py_, px_),   // phi
                     (px_*cos(psi_)*v_ + py_*sin(psi_)*v_)/sqrt(px_*px_ + py_*py_);   // rdot
					 
  }
  
  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < n_sig_; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_radar_, n_z_radar_);
  S.fill(0.);
  for(int col=0;col<n_sig_;col++) {
	z_diff = (Zsig.col(col) - z_pred);

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(col) * z_diff * z_diff.transpose();
  }

  MatrixXd R_lidar_ = MatrixXd(n_z_radar_, n_z_radar_);
  R_lidar_.fill(0.);
  R_lidar_.diagonal() << std_radr_*std_radr_, std_radphi_*std_radphi_, std_radrd_*std_radrd_;

  S += R_lidar_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  Tc.fill(0.);

  //create matrix for Kalman gain
  MatrixXd K = MatrixXd(n_x_, n_z_radar_);  

  // helper vectors
  VectorXd xdiff = VectorXd(n_x_);
  VectorXd zdiff = VectorXd(n_z_radar_);  

  //calculate cross correlation matrix
  for (int col=0; col <n_sig_; col++) {
    xdiff=Xsig_pred_.col(col) - x_;
    //angle normalization
    while (xdiff(3)> M_PI) xdiff(3)-=2.*M_PI;
    while (xdiff(3)<-M_PI) xdiff(3)+=2.*M_PI;

    //zdiff=Zsig.col(col) - z;
	zdiff=Zsig.col(col) - meas_package.raw_measurements_;
	
    //angle normalization
    while (zdiff(1)> M_PI) zdiff(1)-=2.*M_PI;
    while (zdiff(1)<-M_PI) zdiff(1)+=2.*M_PI;
    
    Tc += weights_(col) * xdiff * zdiff.transpose();       
  }

  //calculate Kalman gain K;
  K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  //zdiff = z-z_pred;
  zdiff = meas_package.raw_measurements_ - z_pred;
  
  //angle normalization
  while (zdiff(1)> M_PI) zdiff(1)-=2.*M_PI;
  while (zdiff(1)<-M_PI) zdiff(1)+=2.*M_PI;

  x_ += K*(zdiff);
  P_ -= K*S*K.transpose();

  //Calculate Normalized Innovation Squared (NIS)
  NIS_radar_ = zdiff.transpose() * S.inverse() * zdiff;

#if (TEST==1)
  //std::cout << "x_:\n" << x_ << std::endl;
  //std::cout << "P_:\n" << P_ << std::endl;  
  //std::cin >> n_sig_;
#endif
}
