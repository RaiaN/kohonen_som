using System;
using System.Collections.Generic;

namespace KohonenMap
{
    class Neuron
    {
        public Neuron(int featuresCnt, Random r)
        {
            weights_ = new List<double>(new double[featuresCnt]);
            for (int wi = 0; wi < featuresCnt; ++wi)
            {
                weights_[wi] = r.NextDouble();
            }
        }
        
        public List<double> weights_{ get; set; }
    }
}
