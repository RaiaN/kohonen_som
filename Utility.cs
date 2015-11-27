﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace KohonenMap
{
    //утилитные функции 
    static class Utility
    {
        //метрика Евклида
        public static double euclidMetrics(ref List<double> x, ref List<double> y)
        {
            double proximity = 0.0;
            foreach (var xy in x.Zip(y, (xval, yval) => new { xval = xval, yval = yval }))
            {
                proximity += Math.Pow(Math.Abs(xy.xval - xy.yval), 2);
            }
            return Math.Sqrt(proximity);
        }

        //функция активации (сигмоида)
        public static double calculateProximity(ref List<double> singleObservation, List<double> neuronWeights)
        {
            return Math.Exp(-euclidMetrics(ref singleObservation, ref neuronWeights));
        }

        //считаем близость между наблюдением и весами всех нейронов
        public static double[,] calcProximityMetrics(ref List<double> input, ref Lattice neurons)
        {
            double[,] distances = new double[neurons.getSize(), neurons.getSize()];
            int latticeSize = neurons.getSize();

            for (int rowInd = 0; rowInd < latticeSize; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize; ++colInd)
                {
                    distances[rowInd, colInd] = calculateProximity(ref input, neurons.get(rowInd, colInd).weights_);
                }
            }
            return distances;
        }

        //находим ближайший к наблюдению нейрон используя массив расстояний (между весами нейронов и наблюдением)
        public static Tuple<int, int> findClosestNeuron(ref double[,] distances, int latticeSize)
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

        //Находим ближайший популярный нейрон (не мертвый)
        public static Tuple<int, int> findClosestPopularNeuron(ref double[,] distances, 
                                                               ref SortedDictionary<Tuple<int, int>, List<int>> aliveNeurons, 
                                                               int latticeSize)
        {            
            List<Tuple<double, int, int>> temp = new List<Tuple<double, int, int>>();
            for (int rowInd = 0; rowInd < latticeSize; ++rowInd)
            {
                for (int colInd = 0; colInd < latticeSize; ++colInd)
                {
                    Tuple<int, int> coord = new Tuple<int, int>(rowInd, colInd);
                    //проверяем, что нейрон живой, прежде чем переносить к нему наблюдения от мертвых нейронов
                    if (aliveNeurons.ContainsKey(coord)) 
                    {
                        temp.Add(new Tuple<double, int, int>(distances[rowInd, colInd], rowInd, colInd));
                    }
                }
            }
            //Упорядочиваем нейроны по убыванию популярности (числу наблюдений отнесенных к ним)
            temp = temp.OrderByDescending(i => i.Item1).ToList();
            //берем самый популярный живой нейрон
            return new Tuple<int, int>(temp[0].Item2, temp[0].Item3);
        }

        //Считаем ошибку представления
        public static double calculateTrainingError(ref List<List<double>> winnerWeights, ref List<List<double>> inputData)
        {
            double error = 0.0;
            for (int winnerInd = 0; winnerInd < inputData.Count(); ++winnerInd)
            {
                List<double> weights = winnerWeights[winnerInd];
                List<double> data = inputData[winnerInd];
                double currError = euclidMetrics(ref weights, ref data);
                error += currError * currError;
            }
            return error;
        }

        //Поэлементная разница
        public static List<double> vectorDifference(ref List<double> x, ref List<double> y)
        {
            List<double> diff = new List<double>();
            for (int ind = 0; ind < x.Count; ++ind)
            {
                diff.Add(x[ind] - y[ind]);
            }

            return diff;
        }

        //считаем расстояние между нейронами в сетке
        public static double calcLatticeDist(Tuple<int, int> neuronWinnerCoord, int x, int y)
        {
            List<double> neuronWinner = new List<double>(2) { neuronWinnerCoord.Item1, neuronWinnerCoord.Item2 };
            List<double> other = new List<double>(2) { x, y };
            return euclidMetrics(ref neuronWinner, ref other);
        }

        //инициализируем список списков
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

        //отдельный метод для очистки
        public static void clearLabels(ref List<int>[,] labels, int latticeSize)
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