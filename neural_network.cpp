#pragma once
#include "neural_network.h"



// Neuron Class implementation


double neuron::eta = 0.15;
double neuron::alpha = 0.5;

neuron::neuron(int numOutputs, int myIndex)
{
	for (unsigned i = 0; i < numOutputs; i++)
	{
		m_outputWeights.push_back(connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}
///////////////////


double neuron::transferFunction(double x)
{
	return tanh(x);
}
///////////////////


double neuron::transferFunctionDerivative(double x)
{
	return 1 - x * x;
}
///////////////////


void neuron::feedForward(const layer& previousLayer)
{
	double sum = 0.0;
	for (int i = 0; i < previousLayer.size(); i++)
	{
		sum += previousLayer[i].getOutputVal() *
			previousLayer[i].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = neuron::transferFunction(sum);
}
///////////////////


void neuron::calculateOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * neuron::transferFunctionDerivative(m_outputVal);
}
///////////////////

double neuron::sumDOW(const layer& nextLayer) const
{
	double sum = 0.0;

	for (int n = 0; n < nextLayer.size() - 1; n++)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}
///////////////////

void neuron::calcHiddenGradients(const layer& nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * neuron::transferFunctionDerivative(m_outputVal);
}
///////////////////

void neuron::updateInputWeights(layer& prevLayer)
{
	for (int n = 0; n < prevLayer.size(); n++)
	{
		neuron& Neuron = prevLayer[n];
		double oldDeltaWeight = Neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = eta * Neuron.getOutputVal() * m_gradient
			+ alpha * oldDeltaWeight;

		Neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		Neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;

	}
}




// Neural Network Class implementation


NeuralNetwork::NeuralNetwork(const vector<int>& topology)
{
	int numLayers = topology.size();
	for (int i = 0; i < numLayers; i++)
	{
		m_layers.push_back(layer());
		int numOutputs = (i == numLayers - 1) ? 0 : topology[i + 1];

		for (int j = 0; j <= topology[i]; j++)
		{
			m_layers.back().push_back(neuron(numOutputs, j));

			if (j == topology[i]) {
				cout << "Created a Bias Neuron" << endl;
			}
			else {
				cout << "Created a Neuron" << endl;
			}

		}
		m_layers.back().back().setOutputVal(1.0); //set bias neuron output value = 1.0
	}
}

NeuralNetwork::~NeuralNetwork()
{
}
/////////////////////////////

void NeuralNetwork::ForwardPropagate(const vector<double>& inputVals)
{
	cout << inputVals.size() << " " << m_layers[0].size() - 1 << endl;

	assert(inputVals.size() == m_layers[0].size() - 1);
	//pass input values into the network
	for (int i = 0; i < inputVals.size(); i++)
	{
		//m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//Start forward propagation
	for (int layerNum = 1; layerNum < m_layers.size(); layerNum++)
	{
		layer& previousLayer = m_layers[layerNum - 1];
		for (int neuronNum = 0; neuronNum < m_layers[layerNum].size() - 1; neuronNum++)
		{
			m_layers[layerNum][neuronNum].feedForward(previousLayer);
		}
	}
}
/////////////////////////////

void NeuralNetwork::BackPropagate(const vector<double>& targetVals)
{

	layer& outputLayer = m_layers.back();
	double m_error = 0.0;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;


	//calculae overall net error
	for (int n = 0; n < outputLayer.size() - 1; n++)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1;
	m_error = sqrt(m_error); //RMS 


	//recent average error measure
   // m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
	   // / (m_recentAverageError + 1.0);


	//calcualte output layer gradient
	for (int n = 0; n < outputLayer.size() - 1; n++)
	{
		outputLayer[n].calculateOutputGradients(targetVals[n]);
	}


	//calcualte hidden layer gradient
	for (int layerNum = m_layers.size() - 2; layerNum > 0; layerNum--)
	{
		layer& hiddenLayer = m_layers[layerNum];
		layer& nextLayer = m_layers[layerNum + 1];

		for (int n = 0; n < hiddenLayer.size(); n++)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}


	//Update connection weights
	for (int layerNum = m_layers.size() - 1; layerNum > 0; layerNum--)
	{
		layer& currentLayer = m_layers[layerNum];
		layer& previousLayer = m_layers[layerNum - 1];

		for (int n = 0; n < currentLayer.size() - 1; n++)
		{
			currentLayer[n].updateInputWeights(previousLayer);
		}

	}

}
/////////////////////////////

void NeuralNetwork::GetResults(vector<double> &resultVals)
{
	resultVals.clear();

	for (int n = 0; n < m_layers.back().size() - 1; n++)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}