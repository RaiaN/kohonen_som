using System;
using System.Collections.Generic;

namespace KohonenMap
{
    //Класс для представления нейрона
    class Neuron
    {
        public Neuron(int featuresCnt, Random r)
        {
            //инициализируем веса случайным образом
            weights_ = new List<double>(new double[featuresCnt]);
            for (int wi = 0; wi < featuresCnt; ++wi)
            {
                weights_[wi] = r.NextDouble();
            }
        }

        //веса
        public List<double> weights_{ get; set; }
    }
}
