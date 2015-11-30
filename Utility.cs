using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenMap
{
    static class Utility
    {
        public static double euclidMetrics(List<double> x, List<double> y)
        {
            double proximity = 0.0;
            foreach (var xy in x.Zip(y, (xval, yval) => new { xval = xval, yval = yval }))
            {
                proximity += Math.Pow(Math.Abs(xy.xval - xy.yval), 2);
            }
            return Math.Sqrt(proximity);
        }
        
        public static double calculateProximity(List<double> singleObservation, List<double> neuronWeights)
        {
            return Math.Exp(-euclidMetrics(singleObservation, neuronWeights));
        }
        
        public static double[,] calcProximityMetrics(List<double> input, Lattice neurons)
        {
            double[,] distances = new double[neurons.getSize(), neurons.getSize()];
            int latticeSize = neurons.getSize();

            for (int rowInd = 0; rowInd < latticeSize; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize; ++colInd)
                {
                    distances[rowInd, colInd] = calculateProximity(input, neurons.get(rowInd, colInd).weights_);
                }
            }
            return distances;
        }

        public static Tuple<int, int> neuronIndToCoord(int ni, int latticeSize)
        {
            return new Tuple<int, int>(ni / latticeSize, ni % latticeSize);
        }
        
        public static Tuple<int, int> findClosestNeuron(double[,] distances, int latticeSize)
        {
            double maxProximity = 0.0;
            int maxProximityXCoord = 0,
                maxProximityYCoord = 0;

            for (int rowInd = 0; rowInd < latticeSize; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize; ++colInd)
                {
                    if (distances[rowInd, colInd] > maxProximity)
                    {
                        maxProximityXCoord = rowInd;
                        maxProximityYCoord = colInd;
                        maxProximity = distances[rowInd, colInd];
                    }
                }
            }
            return new Tuple<int, int>(maxProximityXCoord, maxProximityYCoord);
        }
        
        public static Tuple<int, int> findClosestPopularNeuron(double[,] distances, 
                                                               NeuronDict aliveNeurons, 
                                                               int latticeSize)
        {            
            List<Tuple<double, int, int>> temp = new List<Tuple<double, int, int>>();
            for (int rowInd = 0; rowInd < latticeSize; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize; ++colInd)
                {
                    Tuple<int, int> coord = new Tuple<int, int>(rowInd, colInd);
                    if (aliveNeurons.ContainsKey(coord)) 
                    {
                        temp.Add(new Tuple<double, int, int>(distances[rowInd, colInd], rowInd, colInd));
                    }
                }
            }
            temp = temp.OrderByDescending(i => i.Item1).ToList();
            return new Tuple<int, int>(temp[0].Item2, temp[0].Item3);
        }
        
        public static double calculateTrainingError(List<List<double>> winnerWeights, List<List<double>> inputData)
        {
            double error = 0.0;
            for (int winnerInd = 0; winnerInd < inputData.Count(); ++winnerInd)
            {
                List<double> weights = winnerWeights[winnerInd];
                List<double> data = inputData[winnerInd];
                double currError = euclidMetrics(weights, data);
                error += currError * currError;
            }
            return error;
        }
        
        public static List<double> vectorDifference(List<double> x, List<double> y)
        {
            List<double> diff = new List<double>();
            for (int ind = 0; ind < x.Count; ++ind)
            {
                diff.Add(x[ind] - y[ind]);
            }

            return diff;
        }
        
        public static double calcLatticeDist(Tuple<int, int> aCoord, Tuple<int, int> bCoord)
        {
            List<double> aNeuron = new List<double>(2) { aCoord.Item1, aCoord.Item2 };
            List<double> bNeuron = new List<double>(2) { bCoord.Item1, bCoord.Item2 };
            return euclidMetrics(aNeuron, bNeuron);
        }
        
        public static void initLabels(out List<int>[,] labels, int latticeSize)
        {
            labels = new List<int>[latticeSize, latticeSize];
            for (int x = 0; x < latticeSize; ++x)
            {
                for (int y = 0; y < latticeSize; ++y)
                {
                    labels[x, y] = new List<int>();
                }
            }
        }
        
        public static void clearLabels(List<int>[,] labels, int latticeSize)
        {
            for (int x = 0; x < latticeSize; ++x)
            {
                for (int y = 0; y < latticeSize; ++y)
                {
                    labels[x, y].Clear();
                }
            }
        }
    }
}
