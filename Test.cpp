#include <iostream>
#include <vector>
#include <array>
#include "ActivationFunctions.h"
#include <math.h>


template <typename T>
std::ostream& operator<<(std::ostream& obj, const std::vector<T>& A){

    std::string result{""};
    if(A.size()>0){
        result += "[";
        for(int x{}; x<A.size(); x++){
            if(x==A.size()-1){
                result += std::to_string(A[x])+"]";
                continue;
            }
            result += std::to_string(A[x])+", ";
        }
    }
    else{
        return obj << "Vector Empty.";
    }
    
    return obj << result;
}
template <typename T>
std::ostream& operator<<(std::ostream& obj, const std::vector<std::vector<T>>& A){
    obj << "[";
        for(int x{}; x<A.size(); x++){
            if(x==A.size()-1){
                obj << A[x] << "]";
                continue;
            }
            obj << A[x] << ", ";
        }
    return obj;
}

class Neuron{
    public:
        std::vector<double> weights{};
        double bias{1};

        double (*activationFunction)(double){&sigmoid};
        double (*derivativeActivationFunction)(double){&dSigmoid};
        Neuron(std::vector<double> weights, double bias){
            this->weights = weights;
            this->bias = bias;
        }
        Neuron(std::vector<double> weights){
            this->weights = weights;
        }
        Neuron(size_t amount){
            this->weights.reserve(amount);
            for(size_t x{}; x<amount; x++){
                this->weights.push_back(1);
            }
        }
        Neuron(){
            this->weights.push_back(1);
        }
        double feed(double input){
            return activationFunction(weights[0]*input+bias);
        }
        double feed(std::vector<double> input){
            if(input.size() < weights.size()){
                try{
                    throw std::invalid_argument("Not enough inputs.");
                }
                catch(std::invalid_argument a){
                    std::cout << a.what() << "\n";
                    std::cout << "0x" << this << "\n";
                    std::cout << "given inputs: " << input << "\n";
                    std::cout << "Amount of expected inputs: " << weights.size() << "\n";
                    std::cout << "Weights: " << weights << "\n\n";
                }
            }
            double result{};
            for(double x{}; x<weights.size(); x++){
                result += weights[x]*input[x];
            }
            result += bias;
            return activationFunction(result);
        }
        std::vector<double> get_weights(){
            return this->weights;
        }
        double get_weight(int index){
            return this->weights[index];
        }
        void set_weight(double val, int index){
            this->weights[index] = val;
        }
        void set_weights(std::vector<double> val){
            this->weights = val;
        }
        void nudge_weight_by(double val, int index){
            this->weights[index] += val;
        }
        void nudge_weights_by(std::vector<double> val){
            for(int x{}; x<this->weights.size(); x++){
                this->weights[x] += val[x];   
            }
        }
        double get_bias(){
            return this->bias;
        }
        void set_bias(double val){
            this->bias = val;
        }
        void nudge_bias_by(double val){
            this->bias += val;
        }
        void nudge_bias_by(std::vector<double> val){
            this->bias += val[0];
        }
        template <typename CharT, typename Traits>
        std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& obj){
            return obj << "Neuron(" << this << ")";
        }
};

class NeuralNet{
    public:
    std::vector<std::vector<Neuron>> layers{};
    NeuralNet(std::vector<std::vector<Neuron>> layers){
        this->layers = layers;
    }
    NeuralNet(){
        this->layers = {{Neuron({1}), Neuron({1})}, {Neuron({2,2}),Neuron({1,1})}};
        Neuron a1{Neuron({1,1})};
        a1.activationFunction = &relu;
        a1.derivativeActivationFunction = &dRelu;
        this->layers.push_back({a1});
    }
    std::vector<double> feedforward(std::vector<double> input){
        std::vector<double> buffer{input};
        for(std::vector<Neuron> layer : this->layers){
            std::vector<double> layerResults;
            for(Neuron Node : layer){
                layerResults.push_back(Node.feed(buffer));
            }
            buffer = layerResults;
            layerResults.clear();
        }
        return buffer;
    }


    void backpropagate(std::vector<double> input, std::vector<double> expected_result, double learning_rate=0.001){
        std::vector<std::vector<double>> activationHistory{}; // Activation of each Neuron input to output
        for(int x{}; x<this->layers.size(); x++){
            if(x==0){
                activationHistory.push_back(this->feedlayer(input, x)); // Fill Activation of each neuron with correct values
            }
            else{
                activationHistory.push_back(this->feedlayer(activationHistory[x-1], x)); // Fill Activation of each neuron with correct values
            }
        }

        std::vector<std::vector<std::vector<std::vector<double>>>> nudgeVetor{};

        std::vector<std::vector<double>> delCMatrix{}; // OUTPUT TO INPUT

        for(int layer{(int)layers.size()-1}; layer>-1; layer--){
            std::vector<double> delCVector{}; // INPUT TO OUTPUT
            for(int node{}; node<layers[layer].size(); node++){
                if(layer==layers.size()-1){
                    delCVector.push_back(2*(expected_result[node]-activationHistory[layer][node]));
                }
                else{
                    double buffer{};
                    for(int nextLayer{}; nextLayer<layers[layer+1].size(); nextLayer++){
                        
                        double z{layers[layer+1][nextLayer].feed(activationHistory[layer])};
                        
                        buffer += delCMatrix[layers.size()-2-layer][nextLayer] * layers[layer+1][nextLayer].derivativeActivationFunction(z) * layers[layer+1][nextLayer].get_weight(node);
                    }
                    delCVector.push_back(buffer);
                }
            }
            delCMatrix.push_back(delCVector);
            delCVector.clear();
        }
        // 4 layers
        for(int layer{(int)layers.size()-1}; layer >= 0; layer--){ // for every layer
            std::vector<std::vector<std::vector<double>>> layerNudgeVector{};
            for(int node{}; node < this->layers[layer].size(); node++){ // for every node in each layer
                std::vector<std::vector<double>> nodeNudgeVector{};
                
                Neuron neuron {this->layers[layer][node]}; // current Neuron
                double delC{}; // adjustment
                double z{}; //Current neurons output before activation function
                if(layer!=0){ // Set z and sum inputs if no previous layer
                    for(int x{}; x<activationHistory[layer-1].size(); x++){
                        z += activationHistory[layer-1][x]*neuron.get_weight(x);
                    }
                    z += neuron.get_bias(); 
                }
                else{
                    for(int x{}; x<input.size(); x++){
                        z += input[x]*neuron.get_weight(x);
                    }
                    z += neuron.get_bias(); 
                }
                    
                
                // DEL C HERE
                delC = delCMatrix[layers.size()-1-layer][node];
                
                std::vector<double> parameterNudgeVector{};
                for(int weight{}; weight < this->layers[layer][node].get_weights().size(); weight++){
                    double a{}; // activation of a particular neuron in the previous layer or input
                    if(layer!=0){
                        a = activationHistory[layer-1][weight];
                    }
                    else{
                        a = input[weight];
                    }
                    double nudge{};
                    parameterNudgeVector.push_back(a*neuron.derivativeActivationFunction(z)*delC*learning_rate);
                }
                nodeNudgeVector.push_back(parameterNudgeVector);
                parameterNudgeVector.clear();
                //bias nudges here
                parameterNudgeVector.push_back(neuron.derivativeActivationFunction(z)*delC*learning_rate);
                nodeNudgeVector.push_back(parameterNudgeVector);
                layerNudgeVector.push_back(nodeNudgeVector);
            }
            nudgeVetor.push_back(layerNudgeVector);
            layerNudgeVector.clear();
        }
        for(int layer{}; layer<nudgeVetor.size(); layer++){
            for(int node{}; node<nudgeVetor[layer].size(); node++){
                layers[layers.size()-layer-1][node].nudge_weights_by(nudgeVetor[layer][node][0]);
                layers[layers.size()-layer-1][node].nudge_bias_by(nudgeVetor[layer][node][1]);
            }
        }
    }
    
    std::vector<double> feedlayer(std::vector<double> input, int layerIndex){
        std::vector<double> result{};
        for(Neuron x : this->layers[layerIndex]){
            result.push_back(x.feed(input));
        }
        return result;
    }
    double costFunction(std::vector<double> input, std::vector<double> expectedResults){
        std::vector<double> output{this->feedforward(input)};
        if(expectedResults.size() < output.size()){
            return 0;
        }
        double cost{};
        for(int x{}; x<output.size(); x++){
            cost += pow(expectedResults[x]-output[x],2);
        }
        return cost;
    }
    double derivativeCostFunction(std::vector<double> input, std::vector<double> expectedResults){
        std::vector<double> output{this->feedforward(input)};
        if(expectedResults.size() < output.size()){
            return 0;
        }
        double result{};
        for(int x{}; x<output.size(); x++){
            result += 2*(expectedResults[x]-output[x]);
        }
        return result;
    }
    void printInfo(){
        std::cout << "###########################################\n";
        std::cout << "Layers: " << this->layers.size() << "\n";
        for(int layer{1}; layer<=this->layers.size(); layer++){
            std::cout << "Layer " << layer <<":\n";
            std::cout << "\tNode Count: " << this->layers[layer-1].size() <<"\n";
            for(int node{1}; node<=this->layers[layer-1].size(); node++){   
                std::cout << "\t\tNode " << node << ": " << "\n";
                std::cout << "\t\t\tWeights: " << this->layers[layer-1][node-1].get_weights() << "\n";
                std::cout << "\t\t\tBias: " << this->layers[layer-1][node-1].get_bias() << "\n";
            }
        }
        std::cout << "###########################################\n";
    }
};


int main(){
    Neuron a{{1,1}};
    a.activationFunction = &relu;
    a.derivativeActivationFunction = &dRelu;

    NeuralNet test({{Neuron(1),Neuron(1)}, {a}});
    std::cout << test.feedforward({1}) << "\n";
    for(int x{}; x<100; x++){
        test.backpropagate({1}, {1});
        test.backpropagate({2}, {4});
        test.backpropagate({3}, {9});
    }
    std::cout << test.feedforward({1}) << "\n";
    return 0;
}