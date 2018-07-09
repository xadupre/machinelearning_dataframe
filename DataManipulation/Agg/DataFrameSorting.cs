// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Implements sorting functions for dataframe.
    /// </summary>
    public static class DataFrameSorting
    {
        #region typed version

        public static void TSort<T>(IDataFrameView df, ref int[] order, T[] keys, bool ascending)
            where T : IComparable<T>
        {
            if (order == null)
            {
                order = new int[df.Length];
                for (int i = 0; i < order.Length; ++i)
                    order[i] = i;
            }
            if (ascending)
                Array.Sort(order, (x, y) => keys[x].CompareTo(keys[y]));
            else
                Array.Sort(order, (x, y) => -keys[x].CompareTo(keys[y]));
        }

        public static ImmutableTuple<T1>[] TSort<T1>(IDataFrameView df, ref int[] order, IEnumerable<string> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var keys = df.EnumerateItems<T1>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1, T2>[] TSort<T1, T2>(IDataFrameView df, ref int[] order, IEnumerable<string> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var keys = df.EnumerateItems<T1, T2>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1, T2, T3>[] TSort<T1, T2, T3>(IDataFrameView df, ref int[] order, IEnumerable<string> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var keys = df.EnumerateItems<T1, T2, T3>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1>[] TSort<T1>(IDataFrameView df, ref int[] order, IEnumerable<int> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var keys = df.EnumerateItems<T1>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1, T2>[] TSort<T1, T2>(IDataFrameView df, ref int[] order, IEnumerable<int> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var keys = df.EnumerateItems<T1, T2>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        public static ImmutableTuple<T1, T2, T3>[] TSort<T1, T2, T3>(IDataFrameView df, ref int[] order, IEnumerable<int> columns, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var keys = df.EnumerateItems<T1, T2, T3>(columns, ascending).Select(c => c.ToImTuple()).ToArray();
            TSort(df, ref order, keys, ascending);
            return keys;
        }

        #endregion

        #region untyped

        static void RecSort(IDataFrameView df, int[] icols, bool ascending)
        {
            var kind = df.Kinds[icols[0]];
            if (icols.Length == 1)
            {
                switch (kind)
                {
                    case DataKind.BL: df.TSort<DvBool>(icols, ascending); break;
                    case DataKind.I4: df.TSort<DvInt4>(icols, ascending); break;
                    case DataKind.U4: df.TSort<uint>(icols, ascending); break;
                    case DataKind.I8: df.TSort<DvInt8>(icols, ascending); break;
                    case DataKind.R4: df.TSort<float>(icols, ascending); break;
                    case DataKind.R8: df.TSort<double>(icols, ascending); break;
                    case DataKind.TX: df.TSort<DvText>(icols, ascending); break;
                    default:
                        throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                }
            }
            else
            {
                switch (kind)
                {
                    case DataKind.BL: RecSort<DvBool>(df, icols, ascending); break;
                    case DataKind.I4: RecSort<DvInt4>(df, icols, ascending); break;
                    case DataKind.U4: RecSort<uint>(df, icols, ascending); break;
                    case DataKind.I8: RecSort<DvInt8>(df, icols, ascending); break;
                    case DataKind.R4: RecSort<float>(df, icols, ascending); break;
                    case DataKind.R8: RecSort<double>(df, icols, ascending); break;
                    case DataKind.TX: RecSort<DvText>(df, icols, ascending); break;
                    default:
                        throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                }
            }
        }

        static void RecSort<T1>(IDataFrameView df, int[] icols, bool ascending)
            where T1: IEquatable<T1>, IComparable<T1>
        {
            var kind = df.Kinds[icols[1]];
            if (icols.Length == 2)
            {
                switch (kind)
                {
                    case DataKind.BL: df.TSort<T1, DvBool>(icols, ascending); break;
                    case DataKind.I4: df.TSort<T1, DvInt4>(icols, ascending); break;
                    case DataKind.U4: df.TSort<T1, uint>(icols, ascending); break;
                    case DataKind.I8: df.TSort<T1, DvInt8>(icols, ascending); break;
                    case DataKind.R4: df.TSort<T1, float>(icols, ascending); break;
                    case DataKind.R8: df.TSort<T1, double>(icols, ascending); break;
                    case DataKind.TX: df.TSort<T1, DvText>(icols, ascending); break;
                    default:
                        throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                }
            }
            else
            {
                switch (kind)
                {
                    case DataKind.BL: RecSort<T1, DvBool>(df, icols, ascending); break;
                    case DataKind.I4: RecSort<T1, DvInt4>(df, icols, ascending); break;
                    case DataKind.U4: RecSort<T1, uint>(df, icols, ascending); break;
                    case DataKind.I8: RecSort<T1, DvInt8>(df, icols, ascending); break;
                    case DataKind.R4: RecSort<T1, float>(df, icols, ascending); break;
                    case DataKind.R8: RecSort<T1, double>(df, icols, ascending); break;
                    case DataKind.TX: RecSort<T1, DvText>(df, icols, ascending); break;
                    default:
                        throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                }
            }
        }

        static void RecSort<T1, T2>(IDataFrameView df, int[] icols, bool ascending)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var kind = df.Kinds[icols[2]];
            if (icols.Length == 3)
            {
                switch (kind)
                {
                    case DataKind.BL: df.TSort<T1, T2, DvBool>(icols, ascending); break;
                    case DataKind.I4: df.TSort<T1, T2, DvInt4>(icols, ascending); break;
                    case DataKind.U4: df.TSort<T1, T2, uint>(icols, ascending); break;
                    case DataKind.I8: df.TSort<T1, T2, DvInt8>(icols, ascending); break;
                    case DataKind.R4: df.TSort<T1, T2, float>(icols, ascending); break;
                    case DataKind.R8: df.TSort<T1, T2, double>(icols, ascending); break;
                    case DataKind.TX: df.TSort<T1, T2, DvText>(icols, ascending); break;
                    default:
                        throw new NotImplementedException($"Sort is not implemented for type '{kind}'.");
                }
            }
            else
            {
                throw new NotImplementedException($"Sort is not implemented for {icols.Length} columns.");
            }
        }

        public static void Sort(IDataFrameView df, IEnumerable<int> columns, bool ascending = true)
        {
            int[] icols = columns.ToArray();
            RecSort(df, icols, ascending);
        }

        #endregion
    }
}
