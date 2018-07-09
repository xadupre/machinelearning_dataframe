// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// List of implemented aggregated function available after a GroupBy.
    /// </summary>
    public enum AggregatedFunction
    {
        Sum = 1,
        Count = 2,
        Mean = 3,
        Min = 4,
        Max = 5
    }

    /// <summary>
    /// Join strategy.
    /// </summary>
    public enum JoinStrategy
    {
        Inner = 1,
        Left = 2,
        Right = 3,
        Outer = 4
    }

    public enum MultiplyStrategy
    {
        Block = 1,
        Row = 2
    }

    /// <summary>
    /// Interface for a data container held by a dataframe.
    /// </summary>
    public interface IDataContainer
    {
        /// <summary>
        /// Returns a columns based on its position.
        /// </summary>
        IDataColumn GetColumn(int col);

        /// <summary>
        /// Orders the rows.
        /// </summary>
        void Order(int[] order);

        /// <summary>
        /// Reorder columns.
        /// The dataframe is internally modified which means every views
        /// based on it will be probably broken.
        /// </summary>
        void OrderColumns(string[] columns);
    }

    public delegate void GetterAt<DType>(int i, ref DType value);

    /// <summary>
    /// Interface for a column container.
    /// </summary>
    public interface IDataColumn
    {
        /// <summary>
        /// Length of the column
        /// </summary>
        int Length { get; }

        /// <summary>
        /// type of the column 
        /// </summary>
        DataKind Kind { get; }

        /// <summary>
        /// Returns a copy.
        /// </summary>
        IDataColumn Copy();

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        IDataColumn Copy(IEnumerable<int> rows);

        /// <summary>
        /// Creates a new column with the same type but a new length and a constant value.
        /// </summary>
        IDataColumn Create(int n, bool NA = false);

        /// <summary>
        /// Concatenates multiple columns for the same type.
        /// </summary>
        IDataColumn Concat(IEnumerable<IDataColumn> cols);

        /// <summary>
        /// Returns the element at position row
        /// </summary>
        object Get(int row);

        /// <summary>
        /// Get a getter for a specific location.
        /// </summary>
        GetterAt<DType> GetGetterAt<DType>()
            where DType : IEquatable<DType>, IComparable<DType>;

        /// <summary>
        /// Updates value at position row
        /// </summary>
        void Set(int row, object value);

        /// <summary>
        /// Updates all values.
        /// </summary>
        void Set(object value);

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        void Set(IEnumerable<bool> rows, object value);

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        void Set(IEnumerable<int> rows, object value);

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        void Set(IEnumerable<bool> rows, IEnumerable<object> values);

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        void Set(IEnumerable<int> rows, IEnumerable<object> values);

        /// <summary>
        /// The returned getter returns the element
        /// at position <pre>cursor.Position</pre>
        /// </summary>
        ValueGetter<DType> GetGetter<DType>(IRowCursor cursor);

        /// <summary>
        /// exact comparison
        /// </summary>
        bool Equals(IDataColumn col);

        /// <summary>
        /// Returns an enumerator on every row telling if each of them
        /// verfies the condition.
        /// </summary>
        IEnumerable<bool> Filter<TSource>(Func<TSource, bool> predicate);

        /// <summary>
        /// Applies the same function on every value of the column. Example:
        /// <code>
        /// var text = "AA,BB,CC\n0,1,text\n1,1.1,text2";
        /// var df = DataFrame.ReadStr(text);
        /// df["fAA"] = df["AA"].Apply((ref DvInt4 vin, ref float vout) => { vout = (float)vin; });
        /// </code>
        /// </summary>
        NumericColumn Apply<TSrc, TDst>(ValueMapper<TSrc, TDst> mapper)
            where TDst : IEquatable<TDst>, IComparable<TDst>;

        /// <summary>
        /// Sorts the column. Returns the order 
        /// </summary>
        void Sort(ref int[] order, bool ascending = true);
        int[] Sort(bool ascending = true, bool inplace = true);

        /// <summary>
        /// Orders the rows.
        /// </summary>
        void Order(int[] order);

        /// <summary>
        /// Aggregate a column.
        /// </summary>
        TSource Aggregate<TSource>(Func<TSource, TSource, TSource> func, int[] rows = null);

        /// <summary>
        /// Aggregate a column.
        /// </summary>
        TSource Aggregate<TSource>(Func<TSource[], TSource> func, int[] rows = null);

        /// <summary>
        /// Aggregate a column and produces another column.
        /// </summary>
        IDataColumn Aggregate(AggregatedFunction func, int[] rows = null);
    }

    public delegate void MultiGetterAt<DType>(int i, ref DType value);

    /// <summary>
    /// Interface for dataframes and dataframe views.
    /// </summary>
    public interface IDataFrameView : IDataView, IEquatable<IDataFrameView>
    {
        /// <summary>
        /// All rows or all columns.
        /// </summary>
        int[] ALL { get; }

        /// <summary>
        /// Returns the number of rows.
        /// </summary>
        int Length { get; }

        /// <summary>
        /// In case of a DataView, returns the underlying DataFrame.
        /// </summary>
        IDataFrameView Source { get; }

        /// <summary>
        /// In case of a DataView, returns the set of selected columns,
        /// null otherwise.
        /// </summary>
        int[] ColumnsSet { get; }

        /// <summary>
        /// Returns the list of columns.
        /// </summary>
        string[] Columns { get; }

        /// <summary>
        /// Returns the list of types.
        /// </summary>
        DataKind[] Kinds { get; }

        /// <summary>
        /// Returns the number of columns.
        /// </summary>
        int ColumnCount { get; }

        /// <summary>
        /// Returns the shape of the dataframe (number of rows, number of columns).
        /// </summary>
        Tuple<int, int> Shape { get; }

        /// <summary>
        /// Returns a copy of the view.
        /// </summary>
        DataFrame Copy();

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        DataFrame Copy(IEnumerable<int> rows, IEnumerable<int> columns);

        /// <summary>
        /// Sames a GetRowCursor but on a subset of the data.
        /// </summary>
        IRowCursor GetRowCursor(int[] rows, int[] columns, Func<int, bool> needCol, IRandom rand = null);

        /// <summary>
        /// Sames a GetRowCursorSet but on a subset of the data.
        /// </summary>
        IRowCursor[] GetRowCursorSet(int[] rows, int[] columns, out IRowCursorConsolidator consolidator, Func<int, bool> needCol, int n, IRandom rand = null);

        /// <summary>
        /// Retrieves a column by its name.
        /// </summary>
        NumericColumn GetColumn(string colname, int[] rows = null);

        /// <summary>
        /// Returns the column index.
        /// </summary>
        int GetColumnIndex(string name);

        /// <summary>
        /// Retrieves a column by its position.
        /// </summary>
        NumericColumn GetColumn(int col, int[] rows = null);

        /// <summary>
        /// Drops some columns.
        /// Data is not copied.
        /// </summary>
        DataFrameView Drop(IEnumerable<string> colNames);

        #region SQL function

        #region head, tail

        /// <summary>
        /// Returns a view on the first rows.
        /// </summary>
        IDataFrameView Head(int nrows = 5);

        /// <summary>
        /// Returns a view on the last rows.
        /// </summary>
        IDataFrameView Tail(int nrows = 5);

        /// <summary>
        /// Returns a sample.
        /// </summary>
        IDataFrameView Sample(int nrows = 5, bool distinct = false, IRandom rand = null);

        #endregion

        #region select

        /// <summary>
        /// Orders the rows.
        /// </summary>
        void Order(int[] order);

        /// <summary>
        /// Reorder columns.
        /// The dataframe is internally modified which means every views
        /// based on it will be probably broken.
        /// </summary>
        void OrderColumns(string[] columns);

        MultiGetterAt<MutableTuple<T1>> GetMultiGetterAt<T1>(int[] cols)
            where T1 : IEquatable<T1>, IComparable<T1>;
        MultiGetterAt<MutableTuple<T1, T2>> GetMultiGetterAt<T1, T2>(int[] cols)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>;
        MultiGetterAt<MutableTuple<T1, T2, T3>> GetMultiGetterAt<T1, T2, T3>(int[] cols)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>;

        /// <summary>
        /// Enumerates tuples of MutableTuple.
        /// The iterated items are reused.
        /// </summary>
        /// <typeparam name="TTuple">item type</typeparam>
        /// <param name="columns">list of columns to select</param>
        /// <param name="ascending">order</param>
        /// <param name="rows">subset of rows</param>
        /// <returns>enumerator on MutableTuple</returns>
        IEnumerable<TValue> EnumerateItems<TValue>(int[] columns, bool ascending, IEnumerable<int> rows, MultiGetterAt<TValue> getter)
             where TValue : ITUple, new();

        IEnumerable<MutableTuple<T1>> EnumerateItems<T1>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>;
        IEnumerable<MutableTuple<T1, T2>> EnumerateItems<T1, T2>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>;
        IEnumerable<MutableTuple<T1, T2, T3>> EnumerateItems<T1, T2, T3>(IEnumerable<string> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>;

        IEnumerable<MutableTuple<T1>> EnumerateItems<T1>(IEnumerable<int> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>;
        IEnumerable<MutableTuple<T1, T2>> EnumerateItems<T1, T2>(IEnumerable<int> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>;
        IEnumerable<MutableTuple<T1, T2, T3>> EnumerateItems<T1, T2, T3>(IEnumerable<int> columns, bool ascending = true, IEnumerable<int> rows = null)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>;

        #endregion

        #region sort

        /// <summary>
        /// Sorts rows.
        /// </summary>
        void Sort(IEnumerable<string> columns, bool ascending = true);

        /// <summary>
        /// Sorts rows.
        /// </summary>
        void Sort(IEnumerable<int> columns, bool ascending = true);

        void TSort<T1>(IEnumerable<int> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>;
        void TSort<T1, T2>(IEnumerable<int> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>;
        void TSort<T1, T2, T3>(IEnumerable<int> columns, bool ascending = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>;

        #endregion

        #region groupby

        /// <summary>
        /// Aggregates over all rows.
        /// </summary>
        DataFrame Aggregate(AggregatedFunction agg, int[] rows = null, int[] columns = null);

        /// <summary>
        /// Groupby.
        /// </summary>
        IDataFrameViewGroupResults GroupBy(IEnumerable<string> cols, bool sort = true);

        /// <summary>
        /// Groupby.
        /// </summary>
        IDataFrameViewGroupResults GroupBy(IEnumerable<int> cols, bool sort = true);

        DataFrameViewGroupResults<ImmutableTuple<T1>> TGroupBy<T1>(IEnumerable<int> cols, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>;
        DataFrameViewGroupResults<ImmutableTuple<T1, T2>> TGroupBy<T1, T2>(IEnumerable<int> cols, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>;
        DataFrameViewGroupResults<ImmutableTuple<T1, T2, T3>> TGroupBy<T1, T2, T3>(IEnumerable<int> cols, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>;

        #endregion

        #region join

        /// <summary>
        /// Multiplies rows.
        /// </summary>
        IDataFrameView Multiply(int nb, MultiplyStrategy multType = MultiplyStrategy.Block);

        /// <summary>
        /// Join.
        /// </summary>
        DataFrame Join(IDataFrameView right, IEnumerable<string> colsLeft, IEnumerable<string> colsRight,
                       string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true);
        DataFrame Join(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<string> colsRight,
                       string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true);
        DataFrame Join(IDataFrameView right, IEnumerable<string> colsLeft, IEnumerable<int> colsRight,
                       string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true);
        DataFrame Join(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight,
                       string leftSuffix = null, string rightSuffix = null,
                       JoinStrategy joinType = JoinStrategy.Inner, bool sort = true);

        DataFrame TJoin<T1>(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight, string leftSuffix = null, string rightSuffix = null, JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>;
        DataFrame TJoin<T1, T2>(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight, string leftSuffix = null, string rightSuffix = null, JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>;
        DataFrame TJoin<T1, T2, T3>(IDataFrameView right, IEnumerable<int> colsLeft, IEnumerable<int> colsRight, string leftSuffix = null, string rightSuffix = null, JoinStrategy joinType = JoinStrategy.Inner, bool sort = true)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T2 : IEquatable<T2>, IComparable<T2>
            where T3 : IEquatable<T3>, IComparable<T3>;

        #endregion

        #endregion
    }
}
