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
    public static class DataFrameJoining
    {
        #region type version

        public static IEnumerable<DataFrame> TJoin<TKey>(
                                IDataFrameView left, IDataFrameView right,
                                int[] orderLeft, int[] orderRight,
                                TKey[] keysLeft, TKey[] keysRight,
                                int[] icolsLeft, int[] icolsRight,
                                string leftSuffix, string rightSuffix,
                                JoinStrategy joinType,
                                Func<TKey, DataFrameGroupKey[]> funcLeft,
                                Func<TKey, DataFrameGroupKey[]> funcRight)
            where TKey : IEquatable<TKey>, IComparable<TKey>
        {
            var groupLeft = DataFrameGrouping.TGroupBy<TKey>(left, orderLeft, keysLeft, icolsLeft, funcLeft);
            var groupRight = DataFrameGrouping.TGroupBy<TKey>(right, orderRight, keysRight, icolsRight, funcRight);
            var iterLeft = groupLeft.GetEnumerator();
            var iterRight = groupRight.GetEnumerator();
            bool contLeft = iterLeft.MoveNext();
            bool contRight = iterRight.MoveNext();
            leftSuffix = string.IsNullOrEmpty(leftSuffix) ? string.Empty : leftSuffix;
            rightSuffix = string.IsNullOrEmpty(rightSuffix) ? string.Empty : rightSuffix;
            var newColsLeft = left.Columns.Select(c => c + leftSuffix).ToArray();
            var newColsRight = right.Columns.Select(c => c + rightSuffix).ToArray();
            var existsCols = new HashSet<string>(newColsLeft);
            for (int i = 0; i < newColsRight.Length; ++i)
            {
                while (existsCols.Contains(newColsRight[i]))
                    newColsRight[i] += "_y";
                existsCols.Add(newColsRight[i]);
            }
            var newCols = newColsLeft.Concat(newColsRight).ToArray();

            int r;
            while (contLeft || contRight)
            {
                r = contLeft && contRight
                    ? iterLeft.Current.Key.CompareTo(iterRight.Current.Key)
                    : (contRight ? 1 : -1);
                if (r < 0)
                {
                    if (joinType == JoinStrategy.Left || joinType == JoinStrategy.Outer)
                    {
                        var df = iterLeft.Current.Value.Copy();
                        if (!string.IsNullOrEmpty(leftSuffix))
                            df.RenameColumns(newColsLeft);
                        for (int i = 0; i < right.ColumnCount; ++i)
                        {
                            var kind = right.Schema.GetColumnType(i).RawKind;
                            var col = df.AddColumn(newColsRight[i], kind, df.Length);
                            df.GetColumn(col).Set(DataFrameMissingValue.GetMissingValue(kind));
                        }
                        yield return df;
                    }
                    contLeft = iterLeft.MoveNext();
                }
                else if (r > 0)
                {
                    if (joinType == JoinStrategy.Right || joinType == JoinStrategy.Outer)
                    {
                        var df = iterRight.Current.Value.Copy();
                        df.RenameColumns(newColsRight);
                        for (int i = 0; i < left.ColumnCount; ++i)
                        {
                            var kind = left.Schema.GetColumnType(i).RawKind;
                            var col = df.AddColumn(newColsLeft[i], kind, df.Length);
                            df.GetColumn(col).Set(DataFrameMissingValue.GetMissingValue(kind));
                        }
                        df.OrderColumns(newCols);
                        yield return df;
                    }
                    contRight = iterRight.MoveNext();
                }
                else
                {
                    var dfLeft = iterLeft.Current.Value.Copy();
                    var dfRight = iterRight.Current.Value.Copy();
                    if (!string.IsNullOrEmpty(leftSuffix))
                        dfLeft.RenameColumns(newColsLeft);
                    dfRight.RenameColumns(newColsRight);
                    var vleft = dfLeft.Multiply(dfRight.Length, MultiplyStrategy.Block).Copy();
                    var vright = dfRight.Multiply(dfLeft.Length, MultiplyStrategy.Row).Copy();
                    for (int i = 0; i < vright.ColumnCount; ++i)
                        vleft.AddColumn(newColsRight[i], vright.GetColumn(i));
                    yield return vleft;

                    contLeft = iterLeft.MoveNext();
                    contRight = iterRight.MoveNext();
                }
            }
        }

        public static DataFrame TJoin<TMutKey, TImutKey>(
                            IDataFrameView left, IDataFrameView right,
                            int[] rowsLeft, int[] rowsRight,
                            int[] columnsLeft, int[] columnsRight,
                            IEnumerable<int> colsLeft, IEnumerable<int> colsRight,
                            bool sort,
                            string leftSuffix, string rightSuffix,
                            JoinStrategy joinType,
                            MultiGetterAt<TMutKey> getterLeft,
                            MultiGetterAt<TMutKey> getterRight,
                            Func<TMutKey, TImutKey> conv,
                            Func<TImutKey, DataFrameGroupKey[]> convLeft,
                            Func<TImutKey, DataFrameGroupKey[]> convRight)
            where TMutKey : ITUple, new()
            where TImutKey : IComparable<TImutKey>, IEquatable<TImutKey>
        {
            var icolsLeft = colsLeft.ToArray();
            var icolsRight = colsRight.ToArray();
            int[] orderLeft = rowsLeft == null ? rowsLeft.Select(c => c).ToArray() : Enumerable.Range(0, left.Length).ToArray();
            int[] orderRight = rowsLeft == null ? rowsRight.Select(c => c).ToArray() : Enumerable.Range(0, right.Length).ToArray();
            var keysLeft = left.EnumerateItems(icolsLeft, true, rowsLeft, getterLeft).Select(c => conv(c)).ToArray();
            var keysRight = right.EnumerateItems(icolsRight, true, rowsRight, getterRight).Select(c => conv(c)).ToArray();
            if (sort)
            {
                DataFrameSorting.TSort(left, ref orderLeft, keysLeft, true);
                DataFrameSorting.TSort(right, ref orderRight, keysRight, true);
            }
            var iter = TJoin<TImutKey>(left, right, orderLeft, orderRight, keysLeft, keysRight,
                             icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, convLeft, convRight);
            return DataFrame.Concat(iter);
        }

        #endregion

        #region agnostic join

        static DataFrame RecJoin(IDataFrameView left, IDataFrameView right, int[] icolsLeft, int[] icolsRight,
                            string leftSuffix = null, string rightSuffix = null,
                            JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            var kind = left.Kinds[icolsLeft[0]];
            if (icolsLeft.Length == 1)
            {
                switch (kind)
                {
                    case DataKind.BL: return left.TJoin<DvBool>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I4: return left.TJoin<DvInt4>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.U4: return left.TJoin<uint>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I8: return left.TJoin<DvInt8>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R4: return left.TJoin<float>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R8: return left.TJoin<double>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.TX: return left.TJoin<DvText>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    default:
                        throw new NotImplementedException($"Join is not implemented for type '{kind}'.");
                }
            }
            else
            {
                switch (kind)
                {
                    case DataKind.BL: return RecJoin<DvBool>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I4: return RecJoin<DvInt4>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.U4: return RecJoin<uint>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I8: return RecJoin<DvInt8>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R4: return RecJoin<float>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R8: return RecJoin<double>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.TX: return RecJoin<DvText>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    default:
                        throw new NotImplementedException($"Join is not implemented for type '{kind}'.");
                }
            }
        }

        static DataFrame RecJoin<T1>(IDataFrameView left, IDataFrameView right, int[] icolsLeft, int[] icolsRight,
                        string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var kind = left.Kinds[icolsLeft[1]];
            if (icolsLeft.Length == 2)
            {
                switch (kind)
                {
                    case DataKind.BL: return left.TJoin<T1, DvBool>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I4: return left.TJoin<T1, DvInt4>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.U4: return left.TJoin<T1, uint>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I8: return left.TJoin<T1, DvInt8>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R4: return left.TJoin<T1, float>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R8: return left.TJoin<T1, double>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.TX: return left.TJoin<T1, DvText>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    default:
                        throw new NotImplementedException($"Join is not implemented for type '{kind}'.");
                }
            }
            else
            {
                switch (kind)
                {
                    case DataKind.BL: return RecJoin<T1, DvBool>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I4: return RecJoin<T1, DvInt4>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.U4: return RecJoin<T1, uint>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I8: return RecJoin<T1, DvInt8>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R4: return RecJoin<T1, float>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R8: return RecJoin<T1, double>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.TX: return RecJoin<T1, DvText>(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    default:
                        throw new NotImplementedException($"Join is not implemented for type '{kind}'.");
                }
            }
        }

        static DataFrame RecJoin<T1, T2>(IDataFrameView left, IDataFrameView right, int[] icolsLeft, int[] icolsRight,
                        string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var kind = left.Kinds[icolsLeft[2]];
            if (icolsLeft.Length == 3)
            {
                switch (kind)
                {
                    case DataKind.BL: return left.TJoin<T1, T2, DvBool>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I4: return left.TJoin<T1, T2, DvInt4>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.U4: return left.TJoin<T1, T2, uint>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.I8: return left.TJoin<T1, T2, DvInt8>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R4: return left.TJoin<T1, T2, float>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.R8: return left.TJoin<T1, T2, double>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    case DataKind.TX: return left.TJoin<T1, T2, DvText>(right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
                    default:
                        throw new NotImplementedException($"Join is not implemented for type '{kind}'.");
                }
            }
            else
            {
                throw new NotImplementedException($"Join is not implemented for {icolsLeft.Length} columns.");
            }
        }

        public static DataFrame Join(IDataFrameView left, IDataFrameView right,
                        IEnumerable<int> colsLeft, IEnumerable<int> colsRight,
                        string leftSuffix = null, string rightSuffix = null,
                        JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            int[] icolsLeft = colsLeft.ToArray();
            int[] icolsRight = colsRight.ToArray();
            if (icolsRight.Length != icolsLeft.Length)
                throw new DataValueError("Left and right must be joined with the same number of columns.");
            for (int i = 0; i < icolsLeft.Length; ++i)
                if (left.Schema.GetColumnType(icolsLeft[i]) != right.Schema.GetColumnType(icolsRight[i]))
                    throw new DataTypeError("Left and right must be joined with the same number of columns and the same types.");
            return RecJoin(left, right, icolsLeft, icolsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        #endregion
    }
}
