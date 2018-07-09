// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Implements a dense column container.
    /// </summary>
    public class DataColumn<DType> : IDataColumn, IEquatable<DataColumn<DType>>, IEnumerable<DType>
        where DType : IEquatable<DType>, IComparable<DType>
    {
        #region members and easy functions

        /// <summary>
        /// Data for the column.
        /// </summary>
        DType[] _data;

        /// <summary>
        /// Returns a copy.
        /// </summary>
        public IDataColumn Copy()
        {
            var res = new DataColumn<DType>(Length);
            Array.Copy(_data, res._data, Length);
            return res;
        }

        /// <summary>
        /// Returns a copy of a subpart.
        /// </summary>
        public IDataColumn Copy(IEnumerable<int> rows)
        {
            var arows = rows.ToArray();
            var res = new DataColumn<DType>(arows.Length);
            for (int i = 0; i < arows.Length; ++i)
                res._data[i] = _data[arows[i]];
            return res;
        }

        /// <summary>
        /// Creates a new column with the same type but a new length and a constant value.
        /// </summary>
        public IDataColumn Create(int n, bool NA = false)
        {
            var res = new DataColumn<DType>(n);
            if (NA)
            {
                switch (Kind)
                {
                    case DataKind.Bool:
                        res.Set(DvBool.NA);
                        break;
                    case DataKind.I4:
                        res.Set(DvInt4.NA);
                        break;
                    case DataKind.U4:
                        res.Set(0);
                        break;
                    case DataKind.I8:
                        res.Set(DvInt8.NA);
                        break;
                    case DataKind.R4:
                        res.Set(float.NaN);
                        break;
                    case DataKind.R8:
                        res.Set(double.NaN);
                        break;
                    case DataKind.TX:
                        res.Set(DvText.NA);
                        break;
                    default:
                        throw new NotImplementedException($"No missing value convention for type '{Kind}'.");
                }
            }
            return res;
        }

        /// <summary>
        /// Concatenates multiple columns for the same type.
        /// </summary>
        public IDataColumn Concat(IEnumerable<IDataColumn> cols)
        {
            var data = new List<DType>();
            foreach (var col in cols)
            {
                var cast = col as DataColumn<DType>;
                if (cast == null)
                    throw new DataTypeError($"Unable to cast {col.GetType()} in {GetType()}.");
                data.AddRange(cast._data);
            }
            return new DataColumn<DType>(data.ToArray());
        }

        /// <summary>
        /// Number of elements.
        /// </summary>
        public int Length => (_data == null ? 0 : _data.Length);

        /// <summary>
        /// Get a pointer on the raw data.
        /// </summary>
        public DType[] Data => _data;

        public object Get(int row) { return _data[row]; }
        public void Set(int row, object value)
        {
            DType dt;
            ObjectConversion.Convert(ref value, out dt);
            Set(row, dt);
        }

        /// <summary>
        /// Returns type data kind.
        /// </summary>
        public DataKind Kind => SchemaHelper.GetKind<DType>();

        public IEnumerator<DType> GetEnumerator() { foreach (var v in _data) yield return v; }
        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }

        #endregion

        #region constructor

        /// <summary>
        /// Builds the columns.
        /// </summary>
        /// <param name="nb"></param>
        public DataColumn(int nb)
        {
            _data = new DType[nb];
        }

        /// <summary>
        /// Builds the columns.
        /// </summary>
        /// <param name="nb"></param>
        public DataColumn(DType[] data)
        {
            _data = data;
        }

        /// <summary>
        /// Changes the value at a specific row.
        /// </summary>
        public void Set(int row, DType value)
        {
            _data[row] = value;
        }

        /// <summary>
        /// Changes all values.
        /// </summary>
        public void Set(DType value)
        {
            for (int i = 0; i < _data.Length; ++i)
                _data[i] = value;
        }

        /// <summary>
        /// Changes all values.
        /// </summary>
        public void Set(object value)
        {
            var numCol = value as NumericColumn;
            if (numCol is null)
            {
                var enumerable = value as IEnumerable;
                if (enumerable == null || value is string || value is DvText)
                {
                    DType dt;
                    ObjectConversion.Convert(ref value, out dt);
                    for (var row = 0; row < Length; ++row)
                        _data[row] = dt;
                }
                else
                {
                    DType[] dt;
                    ObjectConversion.Convert(ref value, out dt);
                    for (var row = 0; row < Length; ++row)
                        _data[row] = dt[row];
                }
            }
            else
            {
                var arr = numCol.Column as DataColumn<DType>;
                if (arr != null)
                {
                    DType[] dt = arr.Data;
                    for (var row = 0; row < Length; ++row)
                        _data[row] = dt[row];
                }
                else
                {
                    var t = typeof(DataColumn<DType>);
                    throw new DataValueError($"Column oof kind {numCol.Column.Kind} cannot be converted into {t}");
                }
            }
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<bool> rows, object value)
        {
            var irow = 0;
            foreach (var row in rows)
            {
                if (row)
                    Set(irow, value);
                ++irow;
            }
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<int> rows, object value)
        {
            foreach (var row in rows)
                Set(row, value);
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<bool> rows, IEnumerable<object> values)
        {
            var iter = values.GetEnumerator();
            var irow = 0;
            foreach (var row in rows)
            {
                iter.MoveNext();
                if (row)
                    Set(irow, iter.Current);
                ++irow;
            }
        }

        /// <summary>
        /// Updates values based on a condition.
        /// </summary>
        public void Set(IEnumerable<int> rows, IEnumerable<object> values)
        {
            var iter = values.GetEnumerator();
            foreach (var row in rows)
            {
                iter.MoveNext();
                Set(row, iter.Current);
            }
        }

        #endregion

        #region linq

        public IEnumerable<bool> Filter<DType2>(Func<DType2, bool> predicate)
        {
            return (_data as DType2[]).Select(c => predicate(c));
        }

        public int[] Sort(bool ascending = true, bool inplace = true)
        {
            if (inplace)
            {
                Array.Sort(_data);
                if (!ascending)
                    Array.Reverse(_data);
                return null;
            }
            else
            {
                int[] order = null;
                Sort(ref order, ascending);
                return order;
            }
        }

        public void Sort(ref int[] order, bool ascending = true)
        {
            if (order == null)
            {
                order = new int[Length];
                for (int i = 0; i < order.Length; ++i)
                    order[i] = i;
            }

            if (ascending)
                Array.Sort(order, (x, y) => _data[x].CompareTo(_data[y]));
            else
                Array.Sort(order, (x, y) => -_data[x].CompareTo(_data[y]));
        }

        public void Order(int[] order)
        {
            var data = new DType[Length];
            for (int i = 0; i < Length; ++i)
                data[i] = _data[order[i]];
            _data = data;
        }

        public GetterAt<DType2> GetGetterAt<DType2>()
            where DType2 : IEquatable<DType2>, IComparable<DType2>
        {
            var res = GetGetterAtCore() as GetterAt<DType2>;
            if (res == null)
                throw new DataTypeError(string.Format("Type mismatch bytween {0} (expected) and {1} (given).", typeof(DType), typeof(DType2)));
            return res;
        }

        public GetterAt<DType> GetGetterAtCore()
        {
            return (int i, ref DType value) => { value = _data[i]; };
        }

        #endregion

        #region getter and comparison

        /// <summary>
        /// Creates a getter on the column. The getter returns the element at
        /// cursor.Position.
        /// </summary>
        public ValueGetter<DType2> GetGetter<DType2>(IRowCursor cursor)
        {
            var _data2 = _data as DType2[];
            var missing = DataFrameMissingValue.GetMissingValue(Kind);
            return (ref DType2 value) =>
            {
                value = cursor.Position < _data.LongLength
                        ? _data2[cursor.Position]
                        : (DType2)missing;
            };
        }

        public bool Equals(IDataColumn c)
        {
            var obj = c as DataColumn<DType>;
            if (obj == null)
                return false;
            return Equals(obj);
        }

        public bool Equals(DataColumn<DType> c)
        {
            if (Length != c.Length)
                return false;
            for (int i = 0; i < Length; ++i)
                if (!_data[i].Equals(c._data[i]))
                    return false;
            return true;
        }

        #endregion

        #region dataframe functions

        /// <summary>
        /// Applies the same function on every value of the column.
        /// </summary>
        public NumericColumn Apply<TSrc, TDst>(ValueMapper<TSrc, TDst> mapper)
            where TDst : IEquatable<TDst>, IComparable<TDst>
        {
            var maptyped = mapper as ValueMapper<DType, TDst>;
            if (maptyped == null)
                throw new DataValueError("Unexpected input type for this column.");
            var res = new DataColumn<TDst>(Length);
            for (int i = 0; i < res.Length; ++i)
                maptyped(ref Data[i], ref res.Data[i]);
            return new NumericColumn(res);
        }

        public TSource Aggregate<TSource>(Func<TSource, TSource, TSource> func, int[] rows = null)
        {
            var funcTyped = func as Func<DType, DType, DType>;
            if (func == null)
                throw new NotSupportedException($"Type '{typeof(TSource)}' is not compatible with '{typeof(DType)}'.");
            var mapper = GetGenericConverter() as ValueMapper<DType, TSource>;
            var res = AggregateTyped(funcTyped, rows);
            var converted = default(TSource);
            mapper(ref res, ref converted);
            return converted;
        }

        public TSource Aggregate<TSource>(Func<TSource[], TSource> func, int[] rows = null)
        {
            var funcTyped = func as Func<DType[], DType>;
            if (funcTyped == null)
                throw new NotSupportedException($"Type '{typeof(TSource)}' is not compatible with '{typeof(DType)}'.");
            var mapper = GetGenericConverter() as ValueMapper<DType, TSource>;
            var res = AggregateTyped(funcTyped, rows);
            var converted = default(TSource);
            mapper(ref res, ref converted);
            return converted;
        }

        static ValueMapper<DType, DType> GetGenericConverter()
        {
            return (ref DType src, ref DType dst) => { dst = src; };
        }

        public DType AggregateTyped(Func<DType, DType, DType> func, int[] rows = null)
        {
            if (rows == null)
                return _data.Aggregate(func);
            else
                return rows.Select(c => _data[c]).Aggregate(func);
        }

        public DType AggregateTyped(Func<DType[], DType> func, int[] rows = null)
        {
            if (rows == null)
                return func(_data);
            else
                return func(rows.Select(c => _data[c]).ToArray());
        }

        public IDataColumn Aggregate(AggregatedFunction func, int[] rows = null)
        {
            if (typeof(DType) == typeof(DvBool))
                return new DataColumn<DvBool>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(DvBool)), rows) });
            if (typeof(DType) == typeof(DvInt4))
                return new DataColumn<DvInt4>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(DvInt4)), rows) });
            if (typeof(DType) == typeof(uint))
                return new DataColumn<uint>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(uint)), rows) });
            if (typeof(DType) == typeof(DvInt8))
                return new DataColumn<DvInt8>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(DvInt8)), rows) });
            if (typeof(DType) == typeof(float))
                return new DataColumn<float>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(float)), rows) });
            if (typeof(DType) == typeof(double))
                return new DataColumn<double>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(double)), rows) });
            if (typeof(DType) == typeof(DvText))
                return new DataColumn<DvText>(new[] { Aggregate(DataFrameAggFunctions.GetAggFunction(func, default(DvText)), rows) });
            throw new NotImplementedException($"Unkown type '{typeof(DType)}'.");
        }
    }

    #endregion
}
