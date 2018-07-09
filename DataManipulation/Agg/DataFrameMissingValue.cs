// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Ext.DataManipulation
{
    public static class DataFrameMissingValue
    {
        public static object GetMissingValue(DataKind kind, object subcase = null)
        {
            switch (kind)
            {
                case DataKind.BL:
                    return subcase is bool ? (object)(bool)false : DvBool.NA; ;
                case DataKind.I4:
                    return subcase is int ? (object)(int)0 : (object)DvInt4.NA;
                case DataKind.U4:
                    return 0;
                case DataKind.I8:
                    return subcase is Int64 ? (object)(Int64)0 : DvInt8.NA;
                case DataKind.R4:
                    return float.NaN;
                case DataKind.R8:
                    return double.NaN;
                case DataKind.TX:
                    return subcase is string ? (object)(string)null : DvText.NA;
                default:
                    throw new NotImplementedException($"Unknown missing value for type '{kind}'.");
            }
        }
    }
}
