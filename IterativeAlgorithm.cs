using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenMap
{
    class IterativeAlgorithm : Algorithm
    {
        public IterativeAlgorithm(int latticeSize, int featuresCnt) : 
            base(latticeSize, featuresCnt, 
                 epochsNumber: 1000, learningRate: 0.1, tau2: 1000, trainingError: 1e-3) {}
        
        public void run(List<List<double>> inputData, List<List<int>> outputVectors)
        {
            int inputDataLen = inputData.Count();
            List<List<double>> winnerWeights = initWinnerWeights(inputDataLen);

            List<int>[,] labels;
            Utility.initLabels(out labels, latticeSize_);

            int time = 0;
            double currError = 1e3;
            double prevError = 0.0;
            
            while (Math.Abs(currError - prevError) > trainingError_)
            {
                Utility.clearLabels(labels, latticeSize_);
                for (int inputInd = 0; inputInd < inputData.Count(); ++inputInd)
                {
                    List<double> singleObservation = inputData[inputInd];
                    
                    double[,] distances = Utility.calcProximityMetrics(singleObservation, neurons_);
                    Tuple<int, int> neuronWinnerCoord = Utility.findClosestNeuron(distances, latticeSize_);                    
                    winnerWeights[inputInd] = neurons_.get(neuronWinnerCoord).weights_;              
                          
                    updateWeights(singleObservation, neuronWinnerCoord, time);
                    labels[neuronWinnerCoord.Item1, neuronWinnerCoord.Item2].Add(inputInd);
                }
                prevError = currError;
                currError = Utility.calculateTrainingError(winnerWeights, inputData);
                ++time;
            }
        }
        
        private void updateWeights(List<double> input, Tuple<int, int> neuronWinnerCoord, int time)
        {
            for (int rowInd = 0; rowInd < latticeSize_; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize_; ++colInd)
                {
                    int neuronInd = rowInd * latticeSize_ + colInd;
                    List<double> neuronWeights = neurons_.get(rowInd, colInd).weights_;
                    List<double> diff = Utility.vectorDifference(input, neuronWeights);
                    
                    double coef = calcUpdateCoef(neuronWinnerCoord, Utility.neuronIndToCoord(neuronInd, latticeSize_), time);
                    
                    diff = diff.Select(x => x * (-coef)).ToList();
                    List<double> updatedWeights = Utility.vectorDifference(neuronWeights, diff); //value dependens on argument order
                    
                    neurons_.get(rowInd, colInd).weights_ = updatedWeights;
                }
            }
        }
    }
}
