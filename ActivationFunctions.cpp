#include <math.h>
double sigmoid(double x){
    return exp(x)/(exp(x)+1);
}
double dSigmoid(double x){
    return (1-sigmoid(x))*sigmoid(x);
}
double tanh(double x){
    return 2*sigmoid(x/2)-1;
}
double dTanh(double x){
    return 1-tanh(x)*tanh(x);
}
double relu(double x){
    if(x<0){
        return 0;
    }
    return x;
}
double dRelu(double x){
    if(x<0){
        return 0;
    }
    return 1;
}
double leaky_relu(double x){
    if(x<0){
        return x/10;
    }
    return x;
}
double dLeaky_relu(double x){
    if(x<0){
        return .1;
    }
    return 1;
}