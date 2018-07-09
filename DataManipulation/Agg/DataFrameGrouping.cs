// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Implements grouping functions for dataframe.
    /// </summary>
    public static class DataFrameGrouping
    {
        #region type version

        public static IEnumerable<KeyValuePair<TKey, DataFrameViewGroup>> TGroupBy<TKey>(
                                IDataFrameView df, int[] order, TKey[] keys, int[] columns,
                                Func<TKey, DataFrameGroupKey[]> func)
            where TKey : IEquatable<TKey>
        {
            TKey last = keys.Any() ? keys[order[0]] : default(TKey);
            List<int> subrows = new List<int>();
            foreach (var pos in order)
            {
                var cur = keys[pos];
                if (cur.Equals(last))
                    subrows.Add(pos);
                else if (subrows.Any())
                {
                    yield return new KeyValuePair<TKey, DataFrameViewGroup>(last,
                                    new DataFrameViewGroup(func(last), df.Source ?? df, subrows.ToArray(), df.ColumnsSet));
                    subrows.Clear();
                    subrows.Add(pos);
                }
                last = cur;
            }
            if (subrows.Any())
                yield return new KeyValuePair<TKey, DataFrameViewGroup>(last,
                            new DataFrameViewGroup(func(last), df.Source ?? df, subrows.ToArray(), df.ColumnsSet));
        }

        public static DataFrameViewGroupResults<TImutKey> TGroupBy<TMutKey, TImutKey>(
                            IDataFrameView df, int[] rows, int[] columns, IEnumerable<int> cols, bool sort,
                            MultiGetterAt<TMutKey> getter,
                            Func<TMutKey, TImutKey> conv,
                            Func<TImutKey, DataFrameGroupKey[]> conv2)
            where TMutKey : ITUple, new()
            where TImutKey : IComparable<TImutKey>, IEquatable<TImutKey>
        {
            var icols = cols.ToArray();
            int[] order = rows == null ? rows.Select(c => c).ToArray() : Enumerable.Range(0, df.Length).ToArray();
            var keys = df.EnumerateItems(icols, true, rows, getter).Select(c => conv(c)).ToArray();
            if (sort)
                DataFrameSorting.TSort(df, ref order, keys, true);
            var iter = TGroupBy(df, order, keys, columns, conv2);
            return new DataFrameViewGroupResults<TImutKey>(iter);
        }

        #endregion

        #region agnostic groupby

        static IDataFrameViewGroupResults RecGroupBy(IDataFrameView df, int[] icols, bool sort)
        {
            var kind = df.Kinds[icols[0]];
            if (icols.Length == 1)
            {
                switch (kind)
                {
                    case DataKind.BL: return df.TGroupBy<DvBool>(icols, sort);
                    case DataKind.I4: return df.TGroupBy<DvInt4>(icols, sort);
                    case DataKind.U4: return df.TGroupBy<uint>(icols, sort);
                    case DataKind.I8: return df.TGroupBy<DvInt8>(icols, sort);
                    case DataKind.R4: return df.TGroupBy<float>(icols, sort);
                    case DataKind.R8: return df.TGroupBy<double>(icols, sort);
                    case DataKind.TX: return df.TGroupBy<DvText>(icols, sort);
                    default:
                        throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                }
            }
            else
            {
                switch (kind)
                {
                    case DataKind.BL: return RecGroupBy<DvBool>(df, icols, sort);
                    case DataKind.I4: return RecGroupBy<DvInt4>(df, icols, sort);
                    case DataKind.U4: return RecGroupBy<uint>(df, icols, sort);
                    case DataKind.I8: return RecGroupBy<DvInt8>(df, icols, sort);
                    case DataKind.R4: return RecGroupBy<float>(df, icols, sort);
                    case DataKind.R8: return RecGroupBy<double>(df, icols, sort);
                    case DataKind.TX: return RecGroupBy<DvText>(df, icols, sort);
                    default:
                        throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                }
            }
        }

        static IDataFrameViewGroupResults RecGroupBy<T1>(IDataFrameView df, int[] icols, bool sort)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var kind = df.Kinds[icols[1]];
            if (icols.Length == 2)
            {
                switch (kind)
                {
                    case DataKind.BL: return df.TGroupBy<T1, DvBool>(icols, sort);
                    case DataKind.I4: return df.TGroupBy<T1, DvInt4>(icols, sort);
                    case DataKind.U4: return df.TGroupBy<T1, uint>(icols, sort);
                    case DataKind.I8: return df.TGroupBy<T1, DvInt8>(icols, sort);
                    case DataKind.R4: return df.TGroupBy<T1, float>(icols, sort);
                    case DataKind.R8: return df.TGroupBy<T1, double>(icols, sort);
                    case DataKind.TX: return df.TGroupBy<T1, DvText>(icols, sort);
                    default:
                        throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                }
            }
            else
            {
                switch (kind)
                {
                    case DataKind.BL: return RecGroupBy<T1, DvBool>(df, icols, sort);
                    case DataKind.I4: return RecGroupBy<T1, DvInt4>(df, icols, sort);
                    case DataKind.U4: return RecGroupBy<T1, uint>(df, icols, sort);
                    case DataKind.I8: return RecGroupBy<T1, DvInt8>(df, icols, sort);
                    case DataKind.R4: return RecGroupBy<T1, float>(df, icols, sort);
                    case DataKind.R8: return RecGroupBy<T1, double>(df, icols, sort);
                    case DataKind.TX: return RecGroupBy<T1, DvText>(df, icols, sort);
                    default:
                        throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                }
            }
        }

        static IDataFrameViewGroupResults RecGroupBy<T1, T2>(IDataFrameView df, int[] icols, bool sort)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var kind = df.Kinds[icols[2]];
            if (icols.Length == 3)
            {
                switch (kind)
                {
                    case DataKind.BL: return df.TGroupBy<T1, T2, DvBool>(icols, sort);
                    case DataKind.I4: return df.TGroupBy<T1, T2, DvInt4>(icols, sort);
                    case DataKind.U4: return df.TGroupBy<T1, T2, uint>(icols, sort);
                    case DataKind.I8: return df.TGroupBy<T1, T2, DvInt8>(icols, sort);
                    case DataKind.R4: return df.TGroupBy<T1, T2, float>(icols, sort);
                    case DataKind.R8: return df.TGroupBy<T1, T2, double>(icols, sort);
                    case DataKind.TX: return df.TGroupBy<T1, T2, DvText>(icols, sort);
                    default:
                        throw new NotImplementedException($"GroupBy is not implemented for type '{kind}'.");
                }
            }
            else
            {
                throw new NotImplementedException($"soGroupByrt is not implemented for {icols.Length} columns.");
            }
        }

        public static IDataFrameViewGroupResults GroupBy(IDataFrameView df, IEnumerable<int> columns, bool ascending = true)
        {
            int[] icols = columns.ToArray();
            return RecGroupBy(df, icols, ascending);
        }

        #endregion
    }
}
