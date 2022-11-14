#pragma once
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

struct connection
{
	double weight;
	double deltaWeight;
};

class neuron;

typedef vector<neuron> layer;

/////////////////////////////////////////////////////////////////////////////
class neuron
{
public:
	neuron(int numOutputs, int myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }
	void feedForward(const layer& previousLayer);
	void calculateOutputGradients(double targetVal);
	void calcHiddenGradients(const layer& nextLayer);
	void updateInputWeights(layer& prevLayer);

private:
	static double eta; //learning rate [0-1]
	static double alpha; //network momentum [0-1]
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const layer& nextLayer) const;
	double m_outputVal;
	vector<connection> m_outputWeights;
	int m_myIndex;
	double m_gradient;
};



/////////////////////////////////////////////////////////////////////////////
class NeuralNetwork
{
public:
	NeuralNetwork(const vector<int>& topology);
	~NeuralNetwork();

	void ForwardPropagate(const vector<double>& inputVals);
	void BackPropagate(const vector<double>& targetVals);
	void GetResults(vector<double> &resultVals);
	double getRecentAverageError(void) const { return m_recentAverageError; }

private:
	vector<layer> m_layers;
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

