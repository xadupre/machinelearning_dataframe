// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Implements operator for DataFrame for many types.
    /// </summary>
    public static class DataFrameOpMinusHelper
    {
        public const string OperationName = "Minus";

        #region Operation between a column and a value.

        static void Operation<T1, T3>(NumericColumn c1, out T1[] a, out DataColumn<T3> res)
            where T1 : IEquatable<T1>, IComparable<T1>
            where T3 : IEquatable<T3>, IComparable<T3>
        {
            var c1o = c1.Column as DataColumn<T1>;
            if (c1o is null)
                throw new DataTypeError(string.Format("{0} not implemented for type {1}.", OperationName, c1.GetType()));
            res = new DataColumn<T3>(c1.Length);
            a = c1o.Data;
        }

        public static NumericColumn Operation(NumericColumn c1)
        {
            switch (c1.Kind)
            {
                case DataKind.I4:
                    {
                        DvInt4[] a;
                        DataColumn<DvInt4> res;
                        Operation(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, -a[i]);
                        return new NumericColumn(res);
                    }
                case DataKind.I8:
                    {
                        DvInt8[] a;
                        DataColumn<DvInt8> res;
                        Operation(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, -a[i]);
                        return new NumericColumn(res);
                    }
                case DataKind.R4:
                    {
                        float[] a;
                        DataColumn<float> res;
                        Operation(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, -a[i]);
                        return new NumericColumn(res);
                    }
                case DataKind.R8:
                    {
                        double[] a;
                        DataColumn<double> res;
                        Operation(c1, out a, out res);
                        for (int i = 0; i < res.Length; ++i)
                            res.Set(i, -a[i]);
                        return new NumericColumn(res);
                    }
                default:
                    throw new DataTypeError(string.Format("{0} not implemented for column {1}.", OperationName, c1.Kind));
            }
        }

        #endregion
    }
}
