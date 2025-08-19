#ifndef ACTIVATIONFUNC
#define ACTIVATIONFUNC
double relu(double x);
double leaky_relu(double x);
double sigmoid(double x);
// double tanh(double x);
double dRelu(double x);
double dLeaky_relu(double x);
double dSigmoid(double x);
double dTanh(double x);
#endif