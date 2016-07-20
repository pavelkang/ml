/**
 * Logistic Regression
 *
 **/

#include <iostream>
#include <math.h>


class LogisticRegression {
public:
  // data is a matrix of dimension num_data * in_dim
  void read_input(double **data, int num_data, double *label) {
    _data = data;
    _num_data = num_data;
    _label = label;
  }

  // Returns the error
  double train(double error_threshold = 0.0001, double lrate = 0.01, int max_iters = 5000) {
    // initialize the parameters
    memset(_weights, 0, sizeof(double) * _in_dim);

    double last_mrse = 1e10;
    double mrse = 0.0;
    double *prediction = new double[_num_data];
    
    while (max_iters--) {
      mrse = 0.0;
      // for all input data entry, make prediction
      for (int i = 0; i < _num_data; i++) {
	prediction[i] = h(i);
	mrse += pow(prediction[i] - _label[i], 2);
      }
      std::cout << mrse << std::endl;
      // pre-terminate
      if (last_mrse - mrse < error_threshold) {
	return mrse;
      }
      last_mrse = mrse;
      // Update weight
      for (int j = 0; j < _in_dim; j++) {
	double gradient = 0.0;
	for (int i = 0; i < _num_data; i++) {
	  gradient += (prediction[i] - _label[i]) * _data[i][j];
	}
	_weights[j] = _weights[j] - lrate * gradient / _num_data;
      }
    }
    
    delete[] prediction;
    return mrse;
  }
  
  double evaluate(double *in) {
    return prod(in, _weights, _in_dim);
  }

  void show_result() {
    std::cout << "result: ";
    for (int i = 0; i < _in_dim; i++) {
      std::cout << _weights[i] << ", ";
    }
    std::cout << "\n";
  }
  
  LogisticRegression(int in_dim) {
    _in_dim = in_dim;
    _weights = new double[in_dim];
  }
  ~LogisticRegression() {
    delete[] _weights;
  }
  
private:
  // copied from https://github.com/wxwidget/logistic_regression/blob/master/lr.cpp
  // why casing?
  double sigmoid(double x) {
    double e = 2.71828182845904523536;
    if (x >= 10){
      return 1.0 / (1.0 + pow(e, -10));
    }else if (x <= -10){
      return 1.0 / (1.0 + pow(e, 10));
    }
    return 1.0 / (1.0 + pow(e, -x));
  }

  double prod(double *x, double *y, int dim) {
    double result = 0.0;
    for (int i = 0; i < dim; i++) {
      result += x[i] * y[i];
    }
    return result;
  }
  
  double h(int i) {
    return sigmoid(prod(_data[i], _weights, _in_dim));
  }
  
  int _in_dim;
  double **_data;
  int _num_data;
  double *_label;
  double *_weights;
};



int main() {
  int m, n;
  std::cin >> m >> n;
  double **data = new double *[m];
  for (int i = 0; i < m; i++) {
    data[i] = new double[n];
  }
  LogisticRegression model(n);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cin >> data[i][j];
    }
  }
  double *label = new double[n];
  for (int i = 0; i < n; i ++) {
    std::cin >> label[i];
  }
  model.read_input(data, m, label);
  // run training
  model.train();
  // evaluate one by one
  model.show_result();
  return 0;
}
