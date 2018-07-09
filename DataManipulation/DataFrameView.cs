// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    public class DataFrameView : IDataFrameView
    {
        IDataFrameView _src;
        int[] _rows;
        int[] _columns;
        ISchema _schema;

        public int[] ALL { get { return null; } }
        public int Length { get { return _rows == null ? _src.Length : _rows.Length; } }
        public IDataFrameView Source => _src;
        public int[] ColumnsSet => _columns;

        /// <summary>
        /// Initializes a view on a dataframe.
        /// </summary>
        public DataFrameView(IDataFrameView src, IEnumerable<int> rows, IEnumerable<int> columns)
        {
            _src = src;
            _rows = rows == null ? Enumerable.Range(0, src.Length).ToArray() : rows.ToArray();
            _columns = columns == null ? Enumerable.Range(0, src.Schema.ColumnCount).ToArray() : columns.ToArray();
            _schema = new DataFrameViewSchema(src.Schema, _columns);
        }

        #region IDataView API

        public ISchema Schema => _schema;
        public int ColumnCount => _columns == null ? _src.ColumnCount : _columns.Length;

        /// <summary>
        /// Can shuffle the data.
        /// </summary>
        public bool CanShuffle { get { return _src.CanShuffle; } }

        /// <summary>
        /// Returns the number of rows. lazy is unused as the data is stored in memory.
        /// </summary>
        public long? GetRowCount(bool lazy = true)
        {
            return _rows.Length;
        }

        public MultiGetterAt<MutableTuple<T1>> GetMultiGetterAt<T1>(int[] cols)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            var newCols = _columns == null ? cols : cols.Select(c => _columns[c]).ToArray();
            return _src.GetMultiGetterAt<T1>(newCols);
        }

        public MultiGetterAt<MutableTuple<T1, T2>> GetMultiGetterAt<T1, T2>(int[] cols)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            var newCols = _columns == null ? cols : cols.Select(c => _columns[c]).ToArray();
            return _src.GetMultiGetterAt<T1, T2>(newCols);
        }

        public MultiGetterAt<MutableTuple<T1, T2, T3>> GetMultiGetterAt<T1, T2, T3>(int[] cols)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var newCols = _columns == null ? cols : cols.Select(c => _columns[c]).ToArray();
            return _src.GetMultiGetterAt<T1, T2, T3>(newCols);
        }

        /// <summary>
        /// Returns the column index.
        /// </summary>
        public int GetColumnIndex(string name)
        {
            int i;
            if (!Schema.TryGetColumnIndex(name, out i))
                throw new DataNameError($"Unable to find column '{name}'.");
            return i;
        }

        public IRowCursor GetRowCursor(Func<int, bool> needCol, IRandom rand = null)
        {
            return _src.GetRowCursor(_rows, _columns, needCol, rand);
        }

        public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            return _src.GetRowCursorSet(_rows, _columns, out consolidator, needCol, n, rand);
        }

        public IRowCursor GetRowCursor(int[] rows, int[] columns, Func<int, bool> needCol, IRandom rand = null)
        {
            throw Contracts.ExceptNotSupp("Not applicable here, consider building a DataFrameView.");
        }

        public IRowCursor[] GetRowCursorSet(int[] rows, int[] columns, out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null)
        {
            throw Contracts.ExceptNotSupp("Not applicable here, consider building a DataFrameView.");
        }

        /// <summary>
        /// Returns a copy of the view.
        /// </summary>
        public DataFrame Copy()
        {
            return _src.Copy(_rows, _columns);
        }

        /// <summary>
        /// Returns a copy of the view.
        /// </summary>
        public DataFrame Copy(IEnumerable<int> rows, IEnumerable<int> columns)
        {
            var rows2 = rows.Select(i => _rows[i]);
            var columns2 = columns.Select(i => _columns[i]);
            return _src.Copy(rows2, columns2);
        }

        /// <summary>
        /// Converts the data frame into a string.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            using (var stream = new MemoryStream())
            {
                DataFrame.ViewToCsv(this, stream);
                stream.Position = 0;
                using (var reader = new StreamReader(stream))
                    return reader.ReadToEnd().Replace("\r", "").TrimEnd(new char[] { '\n' });
            }
        }

        #endregion

        #region DataFrame

        /// <summary>
        /// Returns the shape of the dataframe (number of rows, number of columns).
        /// </summary>
        public Tuple<int, int> Shape => new Tuple<int, int>(_rows.Length, _columns.Length);

        /// <summary>
        /// Returns the list of columns.
        /// </summary>
        public string[] Columns => _columns == null ? _src.Columns : _columns.Select(c => _src.Columns[c]).ToArray();

        /// <summary>
        /// Returns the list of types.
        /// </summary>
        public DataKind[] Kinds => _columns == null ? _src.Kinds : _columns.Select(c => _src.Kinds[c]).ToArray();

        /// <summary>
        /// A view cannot be modified by adding a column.
        /// </summary>
        public int AddColumn(string name, DataKind kind, int? length)
        {
            throw new DataFrameViewException("A column cannot be added to a DataFrameView.");
        }

        /// <summary>
        /// A view cannot be modified by adding a column.
        /// It must be the same for all columns.
        /// </summary>
        public int AddColumn(string name, IDataColumn values)
        {
            throw new DataFrameViewException("A column cannot be added to a DataFrameView.");
        }

        /// <summary>
        /// Compares two view. First converts them into a DataFrame.
        /// </summary>
        public bool Equals(IDataFrameView dfv)
        {
            return Copy().Equals(dfv.Copy());
        }

        /// <summary>
        /// Drops some columns.
        /// Data is not copied.
        /// </summary>
        public DataFrameView Drop(IEnumerable<string> colNames)
        {
            var idrop = new HashSet<int>(colNames.Select(c => { int col; Schema.TryGetColumnIndex(c, out col); return col; }));
            var ikeep = Enumerable.Range(0, ColumnCount).Where(c => !idrop.Contains(c));
            return new DataFrameView(_src, _rows, ikeep);
        }

        public IEnumerable<TValue> EnumerateItems<TValue>(int[] columns, bool ascending, IEnumerable<int> rows, MultiGetterAt<TValue> getter)
                     where TValue : ITUple, new()
        {
            return _src.EnumerateItems(columns, ascending, rows == null ? _rows : rows.Select(c => _rows[c]), getter);
        }

        public IEnumerable<MutableTuple<T1>> EnumerateItems<T1>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            return _src.EnumerateItems<T1>(columns, ascending, rows == null ? _rows : rows.Select(c => _rows[c]));
        }

        public IEnumerable<MutableTuple<T1, T2>> EnumerateItems<T1, T2>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            return _src.EnumerateItems<T1, T2>(columns, ascending, rows == null ? _rows : rows.Select(c => _rows[c]));
        }

        public IEnumerable<MutableTuple<T1, T2, T3>> EnumerateItems<T1, T2, T3>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
                where T1 : IEquatable<T1>, IComparable<T1>
                where T2 : IEquatable<T2>, IComparable<T2>
                where T3 : IEquatable<T3>, IComparable<T3>
        {
            return _src.EnumerateItems<T1, T2, T3>(columns, ascending, rows == null ? _rows : rows.Select(c => _rows[c]));
        }

        public IEnumerable<MutableTuple<T1>> EnumerateItems<T1>(IEnumerable<int> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            return _src.EnumerateItems<T1>(columns, ascending, rows == null ? _rows : rows.Select(c => _rows[c]));
        }

        public IEnumerable<MutableTuple<T1, T2>> EnumerateItems<T1, T2>(IEnumerable<int> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            return _src.EnumerateItems<T1, T2>(columns, ascending, rows == null ? _rows : rows.Select(c => _rows[c]));
        }

        public IEnumerable<MutableTuple<T1, T2, T3>> EnumerateItems<T1, T2, T3>(IEnumerable<int> columns, bool ascending = true, IEnumerable<int> rows = null)
                where T1 : IEquatable<T1>, IComparable<T1>
                where T2 : IEquatable<T2>, IComparable<T2>
                where T3 : IEquatable<T3>, IComparable<T3>
        {
            return _src.EnumerateItems<T1, T2, T3>(columns, ascending, rows == null ? _rows : rows.Select(c => _rows[c]));
        }

        #endregion

        #region operators

        static int[] ComposeArrayInt(int[] a1, int[] a2)
        {
            var res = new int[a2.Length];
            for (int i = 0; i < a2.Length; ++i)
                res[i] = a1[a2[i]];
            return res;
        }

        /// <summary>
        /// Retrieves a column by its name.
        /// </summary>
        public NumericColumn GetColumn(string colname, int[] rows = null)
        {
            return _src.GetColumn(colname, rows == null ? _rows : ComposeArrayInt(_rows, rows));
        }

        /// <summary>
        /// Retrieves a column by its position.
        /// </summary>
        public NumericColumn GetColumn(int col, int[] rows = null)
        {
            return _src.GetColumn(col, rows == null ? _rows : ComposeArrayInt(_rows, rows));
        }

        /// <summary>
        /// Returns a column.
        /// </summary>
        public NumericColumn this[string colname]
        {
            get { return GetColumn(colname); }
        }

        #endregion

        #region loc / iloc

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public Iloc iloc => new Iloc(this);

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public class Iloc
        {
            DataFrameView _parent;

            public Iloc(DataFrameView parent)
            {
                _parent = parent;
            }

            DataFrame AsDataFrame()
            {
                var parent = _parent._src as DataFrame;
                if (parent is null)
                    throw new NotImplementedException(string.Format("Not implement for type '{0}'.", _parent._src.GetType()));
                return parent;
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[int row, int col]
            {
                get { return AsDataFrame().iloc[_parent._rows[row], _parent._columns[col]]; }
                set { AsDataFrame().iloc[_parent._rows[row], col] = value; }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<bool> rows, int col]
            {
                set { AsDataFrame().iloc[rows, col] = value; }
            }
        }

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public Loc loc => new Loc(this);

        /// <summary>
        /// Artefacts inspired from pandas.
        /// Not necessarily very efficient, it can be used
        /// to modify one value but should not to modify value
        /// in a batch.
        /// </summary>
        public class Loc
        {
            DataFrameView _parent;

            public Loc(DataFrameView parent)
            {
                _parent = parent;
            }

            DataFrame AsDataFrame()
            {
                var parent = _parent._src as DataFrame;
                if (parent is null)
                    throw new NotImplementedException(string.Format("Not implement for type '{0}'.", _parent._src.GetType()));
                return parent;
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[int row, string col]
            {
                get { return AsDataFrame().loc[_parent._rows[row], col]; }
                set { AsDataFrame().loc[_parent._rows[row], col] = value; }
            }

            /// <summary>
            /// Gets or sets elements [i,j].
            /// </summary>
            public object this[string col]
            {
                set { AsDataFrame().loc[col] = value; }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<int> rows, string col]
            {
                get
                {
                    int icol;
                    _parent._src.Schema.TryGetColumnIndex(col, out icol);
                    return new DataFrameView(_parent._src, rows.Select(c => _parent._rows[c]), new[] { icol });
                }
                set { AsDataFrame().loc[rows.Select(c => _parent._rows[c]), col] = value; }
            }

            /// <summary>
            /// Changes the value of a column and a subset of rows.
            /// </summary>
            public object this[IEnumerable<bool> rows, string col]
            {
                set { AsDataFrame().loc[Enumerable.Zip(_parent._rows, rows, (i, b) => b ? -1 : i).Where(c => c >= 0), col] = value; }
            }
        }

        #endregion

        #region SQL function

        #region head, tail, sample

        /// <summary>
        /// Returns a view on the first rows.
        /// </summary>
        public IDataFrameView Head(int nrows = 5)
        {
            nrows = Math.Min(Length, nrows);
            return new DataFrameView(_src, Enumerable.Range(0, nrows).Select(c => _rows[c]).ToArray(), _columns);
        }

        /// <summary>
        /// Returns a view on the last rows.
        /// </summary>
        public IDataFrameView Tail(int nrows = 5)
        {
            nrows = Math.Min(Length, nrows);
            return new DataFrameView(_src, Enumerable.Range(0, nrows).Select(c => c + Length - nrows).Select(c => _rows[c]).ToArray(), _columns);
        }

        /// <summary>
        /// Returns a sample.
        /// </summary>
        public IDataFrameView Sample(int nrows = 5, bool distinct = false, IRandom rand = null)
        {
            nrows = Math.Min(Length, nrows);
            return new DataFrameView(_src, DataFrameRandom.RandomIntegers(nrows, Length, distinct, rand).Select(c => _rows[c]).ToArray(), _columns);
        }

        #endregion  

        #region sort

        /// <summary>
        /// Reorders the rows.
        /// </summary>
        public void Order(int[] order)
        {
            if (_rows == null)
                _rows = order;
            else
            {
                var data = new int[Length];
                for (int i = 0; i < Length; ++i)
                    data[i] = _rows[order[i]];
                _rows = data;
            }
        }

        /// <summary>
        /// Reorders the rows.
        /// </summary>
        public void OrderColumns(string[] columns)
        {
            var cols = columns.Select(c => _src.GetColumnIndex(c)).ToArray();
            _columns = cols;
        }

        /// <summary>
        /// Sorts rows.
        /// </summary>
        public void Sort(IEnumerable<string> columns, bool ascending = true)
        {
            DataFrameSorting.Sort(this, columns.Select(c => GetColumnIndex(c)), ascending);
        }

        /// <summary>
        /// Sorts rows.
        /// </summary>
        public void Sort(IEnumerable<int> columns, bool ascending = true)
        {
            DataFrameSorting.Sort(this, columns, ascending);
        }

        public void TSort<T1>(IEnumerable<int> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            DataFrameSorting.TSort<T1>(this, ref _rows, columns, ascending);
        }

        public void TSort<T1, T2>(IEnumerable<int> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            DataFrameSorting.TSort<T1, T2>(this, ref _rows, columns, ascending);
        }

        public void TSort<T1, T2, T3>(IEnumerable<int> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            DataFrameSorting.TSort<T1, T2, T3>(this, ref _rows, columns, ascending);
        }

        #endregion

        #region groupby

        /// <summary>
        /// Aggregates over all rows.
        /// </summary>
        public DataFrame Aggregate(AggregatedFunction agg, int[] rows = null, int[] columns = null)
        {
            if (rows == null)
                rows = _rows;
            else
                rows = rows.Select(c => _rows[c]).ToArray();
            if (columns == null)
                columns = _columns;
            else
                columns = columns.Select(c => _columns[c]).ToArray();
            return _src.Aggregate(agg, rows, columns);
        }

        /// <summary>
        /// Sum over all rows.
        /// </summary>
        public DataFrame Sum()
        {
            return Aggregate(AggregatedFunction.Sum, _rows, _columns);
        }

        /// <summary>
        /// Min over all rows.
        /// </summary>
        public DataFrame Min()
        {
            return Aggregate(AggregatedFunction.Min, _rows, _columns);
        }

        /// <summary>
        /// Max over all rows.
        /// </summary>
        public DataFrame Max()
        {
            return Aggregate(AggregatedFunction.Max, _rows, _columns);
        }

        /// <summary>
        /// Average over all rows.
        /// </summary>
        public DataFrame Mean()
        {
            return Aggregate(AggregatedFunction.Mean, _rows, _columns);
        }

        /// <summary>
        /// Average over all rows.
        /// </summary>
        public DataFrame Count()
        {
            return Aggregate(AggregatedFunction.Count, _rows, _columns);
        }

        /// <summary>
        /// Groupby.
        /// </summary>
        public IDataFrameViewGroupResults GroupBy(IEnumerable<string> cols, bool sort = true)
        {
            return GroupBy(cols.Select(c => GetColumnIndex(c)), sort);
        }

        /// <summary>
        /// Groupby.
        /// </summary>
        public IDataFrameViewGroupResults GroupBy(IEnumerable<int> cols, bool sort = true)
        {
            return DataFrameGrouping.GroupBy(this, cols, sort);
        }

        public DataFrameViewGroupResults<ImmutableTuple<T1>> TGroupBy<T1>(IEnumerable<int> cols, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            int[] order = _rows.Select(c => c).ToArray();
            var icols = cols.ToArray();
            var scols = icols.Select(c => Schema.GetColumnName(c)).ToArray();
            return DataFrameGrouping.TGroupBy(this, order, _columns, icols, true, GetMultiGetterAt<T1>(icols),
                                              ke => ke.ToImTuple(), ke => DataFrameGroupKey.Create(scols, ke));
        }

        public DataFrameViewGroupResults<ImmutableTuple<T1, T2>> TGroupBy<T1, T2>(IEnumerable<int> cols, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            int[] order = _rows.Select(c => c).ToArray();
            var icols = cols.ToArray();
            var scols = icols.Select(c => Schema.GetColumnName(c)).ToArray();
            return DataFrameGrouping.TGroupBy(this, order, _columns, icols, true, GetMultiGetterAt<T1, T2>(icols),
                                              ke => ke.ToImTuple(), ke => DataFrameGroupKey.Create(scols, ke));
        }

        public DataFrameViewGroupResults<ImmutableTuple<T1, T2, T3>> TGroupBy<T1, T2, T3>(IEnumerable<int> cols, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            int[] order = _rows.Select(c => c).ToArray();
            var icols = cols.ToArray();
            var scols = icols.Select(c => Schema.GetColumnName(c)).ToArray();
            return DataFrameGrouping.TGroupBy(this, order, _columns, icols, true, GetMultiGetterAt<T1, T2, T3>(icols),
                                              ke => ke.ToImTuple(), ke => DataFrameGroupKey.Create(scols, ke));
        }

        #endregion

        #region join

        public IDataFrameView Multiply(int nb, MultiplyStrategy multType = MultiplyStrategy.Block)
        {
            int[] rows = new int[Length * nb];
            switch (multType)
            {
                case MultiplyStrategy.Block:
                    for (int i = 0; i < rows.Length; ++i)
                        rows[i] = _rows[i % Length];
                    break;
                case MultiplyStrategy.Row:
                    for (int i = 0; i < rows.Length; ++i)
                        rows[i] = _rows[i / Length];
                    break;
                default:
                    throw new DataValueError($"Unkown multiplication strategy '{multType}'.");
            }
            return new DataFrameView(_src, rows, _columns);
        }

        /// <summary>
        /// Join.
        /// </summary>
        public DataFrame Join(IDataFrameView right, IEnumerable<string> colsLeft, IEnumerable<string> colsRight,
                       string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            return Join(right, colsLeft.Select(c => GetColumnIndex(c)), colsRight.Select(c => GetColumnIndex(c)), leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame Join(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<string> colsRight,
                       string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            return Join(right, colsLeft, colsRight.Select(c => GetColumnIndex(c)), leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame Join(IDataFrameView right, IEnumerable<string> colsLeft, IEnumerable<int> colsRight,
                           string leftSuffix = null, string rightSuffix = null,
                           JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            return Join(right, colsLeft.Select(c => GetColumnIndex(c)), colsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame Join(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight,
                        string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
        {
            return DataFrameJoining.Join(this, right, colsLeft, colsRight, leftSuffix, rightSuffix, joinType, sort);
        }

        public DataFrame TJoin<T1>(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight, string leftSuffix = null, string rightSuffix = null, JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
        {
            int[] orderLeft = _rows.Select(c => c).ToArray();
            int[] orderRight = (right as DataFrame) is null ? (right as DataFrameView)._rows.Select(c => c).ToArray() : null;
            int[] columnsRight = right.ColumnsSet;
            var icolsLeft = colsLeft.ToArray();
            var icolsRight = colsRight.ToArray();
            var scolsLeft = icolsLeft.Select(c => Schema.GetColumnName(c)).ToArray();
            var scolsRight = icolsRight.Select(c => right.Schema.GetColumnName(c)).ToArray();

            return DataFrameJoining.TJoin(this, right,
                                          orderLeft, orderRight,
                                          _columns, columnsRight,
                                          icolsLeft, icolsRight, sort,
                                          leftSuffix, rightSuffix,
                                          joinType,
                                          GetMultiGetterAt<T1>(icolsLeft),
                                          right.GetMultiGetterAt<T1>(icolsRight),
                                          ke => ke.ToImTuple(),
                                          ke => DataFrameGroupKey.Create(scolsLeft, ke),
                                          ke => DataFrameGroupKey.Create(scolsRight, ke));
        }

        public DataFrame TJoin<T1, T2>(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight, string leftSuffix = null, string rightSuffix = null, JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
        {
            int[] orderLeft = _rows.Select(c => c).ToArray();
            int[] orderRight = (right as DataFrame) is null ? (right as DataFrameView)._rows.Select(c => c).ToArray() : null;
            int[] columnsRight = (right as DataFrame) is null ? (right as DataFrameView)._columns : null;
            var icolsLeft = colsLeft.ToArray();
            var icolsRight = colsRight.ToArray();
            var scolsLeft = icolsLeft.Select(c => Schema.GetColumnName(c)).ToArray();
            var scolsRight = icolsRight.Select(c => right.Schema.GetColumnName(c)).ToArray();

            return DataFrameJoining.TJoin(this, right,
                                          orderLeft, orderRight,
                                          _columns, columnsRight,
                                          icolsLeft, icolsRight, sort,
                                          leftSuffix, rightSuffix,
                                          joinType,
                                          GetMultiGetterAt<T1, T2>(icolsLeft),
                                          right.GetMultiGetterAt<T1, T2>(icolsRight),
                                          ke => ke.ToImTuple(),
                                          ke => DataFrameGroupKey.Create(scolsLeft, ke),
                                          ke => DataFrameGroupKey.Create(scolsRight, ke));
        }

        public DataFrame TJoin<T1, T2, T3>(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight, string leftSuffix = null, string rightSuffix = null, JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            int[] orderLeft = _rows.Select(c => c).ToArray();
            int[] orderRight = (right as DataFrame) is null ? (right as DataFrameView)._rows.Select(c => c).ToArray() : null;
            int[] columnsRight = (right as DataFrame) is null ? (right as DataFrameView)._columns : null;
            var icolsLeft = colsLeft.ToArray();
            var icolsRight = colsRight.ToArray();
            var scolsLeft = icolsLeft.Select(c => Schema.GetColumnName(c)).ToArray();
            var scolsRight = icolsRight.Select(c => right.Schema.GetColumnName(c)).ToArray();

            return DataFrameJoining.TJoin(this, right,
                                          orderLeft, orderRight,
                                          _columns, columnsRight,
                                          icolsLeft, icolsRight, sort,
                                          leftSuffix, rightSuffix,
                                          joinType,
                                          GetMultiGetterAt<T1, T2, T3>(icolsLeft),
                                          right.GetMultiGetterAt<T1, T2, T3>(icolsRight),
                                          ke => ke.ToImTuple(),
                                          ke => DataFrameGroupKey.Create(scolsLeft, ke),
                                          ke => DataFrameGroupKey.Create(scolsRight, ke));
        }

        #endregion

        #endregion
    }
}
