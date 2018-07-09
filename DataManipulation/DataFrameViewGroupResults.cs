// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;


namespace Microsoft.ML.Ext.DataManipulation
{
    public interface IDataFrameViewGroupResults
    {
        DataFrame Count();
        DataFrame Sum();
        DataFrame Min();
        DataFrame Max();
        DataFrame Mean();
    }

    public class DataFrameViewGroupResults<KeyType> : IEnumerable<KeyValuePair<KeyType, DataFrameViewGroup>>, IDataFrameViewGroupResults
    {
        IEnumerable<KeyValuePair<KeyType, DataFrameViewGroup>> _results;

        public DataFrameViewGroupResults(IEnumerable<KeyValuePair<KeyType, DataFrameViewGroup>> results)
        {
            _results = results;
        }

        public IEnumerator<KeyValuePair<KeyType, DataFrameViewGroup>> GetEnumerator() { return _results.GetEnumerator(); }
        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }

        public IEnumerable<DataFrameViewGroup> EnumerateGroups()
        {
            foreach (var pair in _results)
                yield return pair.Value;
        }

        /// <summary>
        /// Aggregates over all rows.
        /// </summary>
        public DataFrame Aggregate(AggregatedFunction func)
        {
            var dfs = new List<DataFrame>();
            int nbkeys = 0;
            foreach (var view in EnumerateGroups())
            {
                var df = view.Drop(view.ColumnsKey);
                var agg = df.Aggregate(func);
                nbkeys = view.Keys.Length;
                foreach (var pair in view.Keys)
                {
                    agg.AddColumn(pair.Key, pair.Kind, 1);
                    agg.loc[0, pair.Key] = pair.Value;
                }
                dfs.Add(agg);
            }
            var res = DataFrame.Concat(dfs);
            var columns = res.Columns;
            int nbnotkeys = columns.Length - nbkeys;
            columns = columns.Skip(nbnotkeys).Concat(columns.Take(nbnotkeys)).ToArray();
            res.OrderColumns(columns);
            return res;
        }

        /// <summary>
        /// Sum over all rows.
        /// </summary>
        public DataFrame Sum()
        {
            return Aggregate(AggregatedFunction.Sum);
        }

        /// <summary>
        /// Min over all rows.
        /// </summary>
        public DataFrame Min()
        {
            return Aggregate(AggregatedFunction.Min);
        }

        /// <summary>
        /// Max over all rows.
        /// </summary>
        public DataFrame Max()
        {
            return Aggregate(AggregatedFunction.Max);
        }

        /// <summary>
        /// Average over all rows.
        /// </summary>
        public DataFrame Mean()
        {
            return Aggregate(AggregatedFunction.Mean);
        }

        /// <summary>
        /// Average over all rows.
        /// </summary>
        public DataFrame Count()
        {
            return Aggregate(AggregatedFunction.Count);
        }
    }
}
