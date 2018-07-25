using System;
using System.Linq;
using Scikit.ML.DataFrame;
using OxyPlot;
using OxyPlot.Series;
using Microsoft.ML.Runtime.Data;

namespace DataFrameExample
{
    class Program
    {
        static void Main(string[] args)
        {
            var df = DataFrame.ReadCsv("airquality.csv");
            Console.WriteLine("Shape: {0}", df.Shape);
            Console.WriteLine("Columns: {0}", string.Join(",", df.Columns));

            Console.WriteLine("df.iloc[0, 1] = {0}", df.iloc[0, 1]);
            Console.WriteLine("df.loc[0, 'Ozone'] = {0}", df.loc[0, "Ozone"]);

            df["Quarter"] = df["Month"] / 4;
            Console.WriteLine("Head\n{0}", df.Head());
            var gr = df.GroupBy(new[] { "Quarter" }).Sum();
            Console.WriteLine("Grouped by Quarter\n{0}", gr);

            var gr2 = df.Drop(new[] { "Ozone", "Solar_R" }).Copy().GroupBy(new[] { "Quarter" }).Sum();
            Console.WriteLine("Grouped by Quarter, no Ozone\n{0}", gr2);

            var plot = new PlotModel { Title = "Simple Graph" };
            var serie = new LineSeries();
            serie.Points.AddRange(Enumerable.Range(0, 2).Select(i => new DataPoint((int)(DvInt4)gr2.iloc[i, 0], (int)(DvInt4)gr2.iloc[i, 1])));

            OxyPlot.Wpf.PngExporter.Export(plot, "graph.png", 600, 400, OxyColors.White);
        }
    }
}
