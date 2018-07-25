using System;
using System.IO;
using System.Linq;
using Scikit.ML.DataFrame;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using Microsoft.ML.Runtime.Data;

namespace GroupBy_Plot
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

            var monthColumn = df["Month"];
            var min = (double)(DvInt4)monthColumn.Aggregate(AggregatedFunction.Min).Get(0);  // Syntax should be improved.
            var max = (double)(DvInt4)monthColumn.Aggregate(AggregatedFunction.Max).Get(0);  // Syntax should be improved.
            df["MonthFrac"] = (df["Month"] - min) / (max - min);
            Console.WriteLine("Head\n{0}", df.Head());
            var gr = df.GroupBy(new[] { "MonthFrac" }).Sum();
            Console.WriteLine("Grouped by MonthFrac\n{0}", gr);

            var gr2 = df.Drop(new[] { "Ozone", "Solar_R" }).Copy().GroupBy(new[] { "MonthFrac" }).Sum();
            Console.WriteLine("Grouped by MonthFrac, no Ozone\n{0}", gr2);

            var plot = new PlotModel { Title = "Simple Graph" };
            var serie = new ScatterSeries();
            serie.Points.AddRange(Enumerable.Range(0, gr.Shape.Item1).Select(i => new ScatterPoint((double)gr2.iloc[i, 0], (float)gr2.iloc[i, 2])));
            plot.Series.Add(serie);
            plot.Axes.Add(new LinearAxis { Position = AxisPosition.Bottom, Title = gr2.Columns[0] });
            plot.Axes.Add(new LinearAxis { Position = AxisPosition.Left, Title = gr2.Columns[2] });

            var plotString = SvgExporter.ExportToString(plot, 600, 400, true);
            File.WriteAllText("graph2.svg", plotString);
        }
    }
}
