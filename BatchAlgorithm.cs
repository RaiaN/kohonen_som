using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenMap
{
    class NeuronDict : SortedDictionary<Tuple<int, int>, List<int>>
    { }
    
    class BatchAlgorithm : Algorithm
    {
        public BatchAlgorithm(int latticeSize, int featuresCnt) : 
            base(latticeSize, featuresCnt, 
                 epochsNumber: 1000, learningRate: 0.1, tau2: 1000, trainingError: 1e-3) { }
        
        public void run(ref List<List<double>> inputData, ref List<List<int>> outputVectors)
        {
            int inputDataLen = inputData.Count();
            List<List<double>> winnerWeights = initWinnerWeights(inputDataLen);

            List<int>[,] labels;
            Utility.initLabels(out labels, latticeSize_);
            
            double[,] latticeDist = precalcLatticeDist();

            int time = 0;
            double currError = 1e3;
            double prevError = 0.0;            
            while (Math.Abs(currError - prevError) > trainingError_)
            {
                Utility.clearLabels(ref labels, latticeSize_);
                for (int inputInd = 0; inputInd < inputData.Count(); ++inputInd)
                {
                    List<double> singleObservation = inputData[inputInd];
                    double[,] distances = Utility.calcProximityMetrics(ref singleObservation, ref neurons_);                    
                    Tuple<int, int> neuronWinnerCoord = Utility.findClosestNeuron(ref distances, latticeSize_);                    
                    winnerWeights[inputInd] = neurons_.get(neuronWinnerCoord).weights_;
                    labels[neuronWinnerCoord.Item1, neuronWinnerCoord.Item2].Add(inputInd);
                }
                updateWeights(ref labels, ref inputData, ref latticeDist, time);
                
                prevError = currError;
                currError = Utility.calculateTrainingError(ref winnerWeights, ref inputData);
                ++time;
            }

            NeuronDict aliveNeurons = chooseAliveNeurons(ref labels);
            spreadOverPopularNeurons(ref inputData, ref labels, ref aliveNeurons, ref inputData);
            showInfo(ref aliveNeurons, ref outputVectors);           
            
            //store utilities
            //storeLatticeWeights("ba_weights_" + time.ToString() + ".txt");
            //currError = Utility.calculateTrainingError(ref winnerWeights, ref inputData);
            //Console.WriteLine("Training error AFTER exlusion dead nodes: " + currError);
            //Console.WriteLine("Epochs passed: " + time);
        }
        
        private double[,] precalcLatticeDist()
        {
            int neuronsNumber = latticeSize_ * latticeSize_;
            double[,] latticeDist = new double[neuronsNumber, neuronsNumber];
            for (int n1 = 0; n1 < neuronsNumber; ++n1)
            {
                for (int n2 = n1; n2 < neuronsNumber; ++n2)
                {
                    Tuple<int, int> coord = new Tuple<int, int>((int)n1 / latticeSize_, n1 % latticeSize_);
                    latticeDist[n1, n2] = Utility.calcLatticeDist(coord, (int)n2 / latticeSize_, n2 % latticeSize_);
                    latticeDist[n2, n1] = latticeDist[n1, n2];
                }
            }
            return latticeDist;
        }
        
        private List<double> calcMeanVector(ref List<int> labels, ref List<List<double>> inputData)
        {
            int labelsLen = labels.Count();
            List<double> meanVector = new List<double>(new double[featuresCnt_]);
            meanVector = meanVector.Select(x => 0.0).ToList();
            if (labelsLen == 0)
            {
                return meanVector;
            }
          
            foreach (int label in labels)
            {
                meanVector = meanVector.Zip(inputData[label], (x, y) => x + y).ToList();
            }
            meanVector = meanVector.Select(x => x / (1.0 * labelsLen)).ToList();
            return meanVector;
        }
        
        private List<Tuple<int, int>> getNeighbourhood(ref double[,] latticeDist, int neuronNumber, double delta)
        {
            int neuronsNumber = latticeSize_ * latticeSize_;
            List<Tuple<int, int>> nb = new List<Tuple<int, int>>();
            for (int ni = 0; ni < neuronsNumber; ++ni)
            {
                if (latticeDist[neuronNumber, ni] <= delta)
                {
                    nb.Add(new Tuple<int, int>((int)ni / latticeSize_, ni % latticeSize_));
                }                
            }
            return nb;
        }
        
        private void calculateUpdatedWeights(ref double[,] latticeDist, 
                                             ref List<int>[,] labels, 
                                             ref List<List<double>> inputData, 
                                             int neuronNumber, int time, int rowInd, int colInd,
                                             ref List<double> numerator, ref double denominator)
        {
            double delta = width_ * Math.Exp(-time / tau1_);            
            List<Tuple<int, int>> nb = getNeighbourhood(ref latticeDist, neuronNumber, delta);
            
            double denominatorPart = 0.0;
            foreach (Tuple<int, int> n in nb)
            {
                int x = n.Item1;
                int y = n.Item2;
                List<int> currLabels = labels[x, y];
                
                double updateCoef = calcUpdateCoef(new Tuple<int, int>(x, y), rowInd, colInd, time) * currLabels.Count();                
                List<double> currNumerator = calcMeanVector(ref currLabels, ref inputData).Select(val => val * updateCoef).ToList();

                numerator = numerator.Zip(currNumerator, (v1, v2) => v1 + v2).ToList();
                denominatorPart += updateCoef;
            }
            denominator += denominatorPart;
            
            if (Math.Abs(denominator) > 0)
            {
                neurons_.get(rowInd, colInd).weights_ = numerator.Select(x => x / (denominatorPart * 1.0)).ToList();
            }
        }
        
        private void updateWeights(ref List<int>[,] labels, 
                                   ref List<List<double>> inputData, 
                                   ref double[,] latticeDist,
                                   int time)
        {           
            for (int rowInd = 0; rowInd < latticeSize_; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize_; ++colInd)
                {                                        
                    int neuronNumber = rowInd * latticeSize_ + colInd;
                    List<double> numerator = new List<double>(new double[featuresCnt_]);
                    numerator = numerator.Select(x => 0.0).ToList();
                    double denominator = 0.0;

                    calculateUpdatedWeights(ref latticeDist, ref labels, ref inputData, neuronNumber, time, rowInd, colInd,
                                            ref numerator, ref denominator);
                }
            }
        }

        private NeuronDict chooseAliveNeurons(ref List<int>[,] labels)
        {
            NeuronDict aliveNeurons = new NeuronDict();
            for (int x = 0; x < latticeSize_; ++x)
            {
                for (int y = 0; y < latticeSize_; ++y)
                {
                    List<int> ls = labels[x, y];
                    int labelsLen = ls.Count();

                    if (labelsLen > 2)
                    {
                        aliveNeurons[new Tuple<int, int>(x, y)] = labels[x, y].Select(val => val).ToList();
                        labels[x, y].Clear();
                    }
                }
            }
            return aliveNeurons;
        }

        private void spreadOverPopularNeurons(ref List<List<double>> inputData, 
                                              ref List<int>[,] labels, 
                                              ref NeuronDict aliveNeurons,
                                              ref List<List<double>> winnerWeights)
        {
            for (int x = 0; x < latticeSize_; ++x)
            {
                for (int y = 0; y < latticeSize_; ++y)
                {
                    List<int> ls = labels[x, y];
                    int labelsLen = ls.Count();

                    if (labelsLen > 0 && labelsLen <= 2) //just check "introvert" neurons
                    {
                        foreach (int label in ls)
                        {
                            List<double> singleObservation = inputData[label];

                            double[,] distances = Utility.calcProximityMetrics(ref singleObservation, ref neurons_);
                            Tuple<int, int> closestPopularNeuron = Utility.findClosestPopularNeuron(ref distances, 
                                                                                                    ref aliveNeurons, 
                                                                                                    latticeSize_);

                            aliveNeurons[closestPopularNeuron].Add(label);
                            winnerWeights[label] = neurons_.get(closestPopularNeuron).weights_;
                        }
                    }
                }
            }
        }

        private void showInfo(ref NeuronDict aliveNeurons, ref List<List<int>> outputVectors)
        {
            List<KeyValuePair<Tuple<int, int>, List<int>>> temp = aliveNeurons.ToList();
            temp.Sort((x, y) =>
            {
                return -1 * x.Value.Count().CompareTo(y.Value.Count());
            });

            Console.WriteLine("Top 3 neuron weights by popularity:");
            for (int pind = 0; pind < 3; ++pind)
            {
                Console.WriteLine(String.Join(" ", neurons_.get(temp[pind].Key).weights_));
                int classesCount = outputVectors[pind].Count();
                List<double> classes = new List<double>(new double[classesCount]);
                foreach (int inputInd in temp[pind].Value)
                {
                    classes = classes.Zip(outputVectors[inputInd], (x, y) => (x + y)).ToList();
                }
                classes = classes.Select(x => 100.0 * x / (1.0 * temp[pind].Value.Count())).ToList();
                Console.WriteLine("Class representation: " + String.Join("% ", classes) + "%\n");
            }
        }
    }
}
