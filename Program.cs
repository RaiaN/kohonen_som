using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using System.Diagnostics;

namespace KohonenMap
{
    class Program
    {
        static void readData(List<List<double>> inputData, List<List<int>> outputVectors, out int featuresCnt, out int outputLen)
        {
            using (System.IO.StreamReader sr = new System.IO.StreamReader("data/cancer1.dt"))
            {
                string line;
                line = sr.ReadLine();
                List<int> inputParams = line.Split(' ').Select(x => Int32.Parse(x)).ToList();
                featuresCnt = inputParams[0];
                outputLen = inputParams[1];

                var style = NumberStyles.Float | NumberStyles.AllowThousands | NumberStyles.AllowDecimalPoint;
                var culture = CultureInfo.InvariantCulture;
                while ((line = sr.ReadLine()) != null)
                {
                    List<double> rowData = line.Split(' ').Select(x => double.Parse(x, style, culture)).ToList();
                    inputData.Add(rowData.GetRange(0, featuresCnt));
                    List<int> outputVector = rowData.GetRange(featuresCnt, outputLen).Select(x => (int)x).ToList();
                    outputVectors.Add(outputVector);
                }
            }
        }
        
        static double calcMean(IEnumerable<double> values)
        {
            return values.Average();
        }
        
        static double calcStdDev(IEnumerable<double> values)
        {
            double ret = 0;   
            double avg = values.Average();     
            double sum = values.Sum(d => Math.Pow(d - avg, 2));
            ret = Math.Sqrt((sum) / (values.Count() - 1));
            return ret;
        }

        static void run(int latticeSize)
        {
            int ITERATIONS = 1;
            var culture = CultureInfo.InvariantCulture; 
            Stopwatch sw;
            
            int featuresCnt;
            int outputLen;

            List<List<double>> inputData = new List<List<double>>();
            List<List<int>> outputVectors = new List<List<int>>();
            readData(inputData, outputVectors, out featuresCnt, out outputLen);
            
            List<double> executionTime = new List<double>();
            
            //ITERATIVE
            /*Console.WriteLine("Iterative learning:");           
            for (int repeat = 0; repeat < ITERATIONS; ++repeat)
            {
                sw = Stopwatch.StartNew();
                IterativeAlgorithm ia = new IterativeAlgorithm(latticeSize, featuresCnt);
                ia.run(ref inputData, ref outputVectors);
                sw.Stop();
                Console.WriteLine("Execution time: {0}ms", sw.Elapsed.TotalMilliseconds);
                executionTime.Add(sw.Elapsed.TotalMilliseconds);
            }
            executionTime.RemoveAt(0);
            
            double mean_ia = calcMean(executionTime);
            double std_ia = calcStdDev(executionTime);
            double l_ia = Math.Max(0, Math.Round((mean_ia - std_ia), 3));
            double r_ia = Math.Round((mean_ia + std), 3);
            Console.WriteLine(l_ia.ToString(culture) + ", " + r_ia.ToString(culture));*/

            executionTime.Clear();
            
            Console.WriteLine("\nBatch learning:");
            for (int repeat = 0; repeat < ITERATIONS; ++repeat)
            {
                sw = Stopwatch.StartNew();
                BatchAlgorithm ba = new BatchAlgorithm(latticeSize, featuresCnt);
                ba.run(inputData, outputVectors);
                sw.Stop();
                Console.WriteLine("Execution time: {0}ms", sw.Elapsed.TotalMilliseconds);
                executionTime.Add(sw.Elapsed.TotalMilliseconds);
            }
            executionTime.RemoveAt(0);

            /*double mean_ba = calcMean(executionTime);
            double std_ba = calcStdDev(executionTime);
            double l_ba = Math.Max(0, Math.Round((mean_ba - std_ba), 3));
            double r_ba = Math.Round((mean_ba + std_ba), 3);
            Console.WriteLine(l_ba.ToString(culture) + ", " + r_ba.ToString(culture));*/
        }

        static void Main(string[] args)
        {
            int latticeSize = 3; //Convert.ToInt32(args[1]);
            run(latticeSize);
            Console.WriteLine("Done!");
            Console.ReadKey();
        }
    }
}
