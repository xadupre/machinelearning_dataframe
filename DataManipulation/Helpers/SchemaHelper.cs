// See the LICENSE file in the project root for more information.

using System;
using System.Text;
using Data = Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data.Conversion;


namespace Scikit.ML.DataFrame
{
    /// <summary>
    /// Helpers about ISchema.
    /// </summary>
    public static class SchemaHelper
    {
        public static ColumnType DataKind2ColumnType(DataKind kind, IChannel ch = null)
        {
            switch (kind)
            {
                case DataKind.BL:
                    return BoolType.Instance;
                case DataKind.R4:
                    return NumberType.R4;
                case DataKind.R8:
                    return NumberType.R8;
                case DataKind.I1:
                    return NumberType.I1;
                case DataKind.I2:
                    return NumberType.I2;
                case DataKind.I4:
                    return NumberType.I4;
                case DataKind.I8:
                    return NumberType.I8;
                case DataKind.U1:
                    return NumberType.U1;
                case DataKind.U2:
                    return NumberType.U2;
                case DataKind.U4:
                    return NumberType.U4;
                case DataKind.U8:
                    return NumberType.U8;
                case DataKind.Text:
                    return TextType.Instance;
                case DataKind.U16:
                    return NumberType.UG;
                case DataKind.DateTime:
                    return DateTimeType.Instance;
                case DataKind.DateTimeZone:
                    return DateTimeZoneType.Instance;
                default:
                    throw ch != null ? ch.Except("Unknown kind {0}", kind) : Contracts.Except("Unknown kind {0}", kind);
            }
        }

        /// <summary>
        /// Convert a type description into another type description.
        /// </summary>

        public static ColumnType Convert(TextLoader.Column col, IChannel ch = null)
        {
            if (!col.Type.HasValue)
                throw ch != null ? ch.Except("Kind is null") : Contracts.Except("kind is null");
            if (col.Source != null && col.Source.Length > 0)
            {
                if (col.Source.Length != 1)
                    throw Contracts.ExceptNotImpl("Convert of TextLoader.Column is not implemented for more than on range.");
                if (col.Source[0].ForceVector)
                {
                    if (!col.Source[0].Max.HasValue)
                        throw ch != null ? ch.Except("A vector column needs a dimension")
                                         : Contracts.Except("A vector column needs a dimension");
                    int delta = col.Source[0].Max.Value - col.Source[0].Min + 1;
                    var colType = DataKind2ColumnType(col.Type.Value, ch);
                    return new VectorType(colType.AsPrimitive, delta);
                }
            }
            if (col.KeyRange != null)
            {
                var r = col.KeyRange;
                return new KeyType(col.Type.Value, r.Min,
                                    r.Max.HasValue ? (int)(r.Max.Value - r.Min + 1) : 0,
                                    r.Contiguous);
            }
            else
                return DataKind2ColumnType(col.Type.Value, ch);
        }

        /// <summary>
        /// To display a schema.
        /// </summary>
        /// <param name="schema">schema</param>
        /// <param name="sep">column separator</param>
        /// <param name="vectorVec">if true, show Vec<R4, 2> and false, shows :R4:5-6 </R4> does the same for keys</param>
        /// <param name="keepHidden">keepHidden columns?</param>
        /// <returns>schema as a string</returns>
        public static string ToString(ISchema schema, string sep = "; ", bool vectorVec = true, bool keepHidden = false)
        {
            var builder = new StringBuilder();
            string name, type;
            string si;
            int lag = 0;
            for (int i = 0; i < schema.ColumnCount; ++i)
            {
                if (!keepHidden && schema.IsHidden(i))
                    continue;
                if (builder.Length > 0)
                    builder.Append(sep);
                name = schema.GetColumnName(i);
                var t = schema.GetColumnType(i);
                if (vectorVec || (!t.IsVector && !t.IsKey))
                {
                    type = schema.GetColumnType(i).ToString().Replace(" ", "");
                    si = (i + lag).ToString();
                }
                else
                {
                    if (t.IsVector)
                    {
                        if (t.AsVector.DimCount != 1)
                            throw Contracts.ExceptNotSupp("Only vector with one dimension are supported.");
                        type = t.ItemType.RawKind.ToString();
                        si = string.Format("{0}-{1}", i + lag, i + lag + t.AsVector.GetDim(0) - 1);
                        lag += t.AsVector.GetDim(0) - 1;
                    }
                    else if (t.IsKey && t.AsKey.Contiguous)
                    {
                        var k = t.AsKey;
                        type = k.Count > 0
                                    ? string.Format("{0}[{1}-{2}]", k.RawKind, k.Min, k.Min + (ulong)k.Count - 1)
                                    : string.Format("{0}[{1}-{2}]", k.RawKind, k.Min, "*");
                        si = i.ToString();
                    }
                    else
                        throw Contracts.ExceptNotImpl(string.Format("Unable to process type '{0}'.", t));
                }

                builder.Append(string.Format("{0}:{1}:{2}", name, type, si));
            }
            return builder.ToString();
        }

        public static void CheckSchema(IHostEnvironment host, ISchema sch1, ISchema sch2)
        {
            if (sch1.ColumnCount != sch2.ColumnCount)
                throw host.Except("Mismatch between input schema and cached schema #columns {0} != # cached columns {1}.\nSchema 1:{2}\nSchema 2: {3}",
                    sch1.ColumnCount, sch2.ColumnCount, ToString(sch1), ToString(sch2));
            for (int i = 0; i < sch2.ColumnCount; ++i)
            {
                if (sch1.GetColumnName(i) != sch2.GetColumnName(i))
                    throw host.Except("Name mismatch at column {0}: '{1}' != '{2}'.\nSchema 1:{3}\nSchema 2: {4}", i,
                        sch1.GetColumnName(i), sch2.GetColumnName(i), ToString(sch1), ToString(sch2));
                if (!sch1.GetColumnType(i).Equals(sch2.GetColumnType(i)))
                    throw host.Except("Type mismatch at column {0}: '{1}' != '{2}'.\nSchema 1:{3}\nSchema 2: {4}", i,
                        sch1.GetColumnType(i), sch2.GetColumnType(i), ToString(sch1), ToString(sch2));
            }
        }

        public static bool CompareSchema(ISchema sch1, ISchema sch2, bool raise = false)
        {
            if (sch1.ColumnCount != sch2.ColumnCount)
            {
                if (raise)
                    throw Contracts.Except("Different number of columns {0} != {1}\nS1: {2}\nS2: {3}",
                        sch1.ColumnCount, sch2.ColumnCount,
                        ToString(sch1), ToString(sch2));
                else
                    return false;
            }
            for (int i = 0; i < sch1.ColumnCount; ++i)
            {
                if (sch1.GetColumnName(i) != sch2.GetColumnName(i))
                {
                    if (raise)
                        throw Contracts.Except("Column name {0} is different {1} != {2}\nS1: {3}\nS2: {4}",
                                i, sch1.GetColumnName(i), sch2.GetColumnName(i),
                                ToString(sch1), ToString(sch2));
                    else
                        return false;
                }
                if (sch1.GetColumnType(i) != sch2.GetColumnType(i))
                {
                    var t1 = sch1.GetColumnType(i);
                    var t2 = sch2.GetColumnType(i);
                    bool r = t1 != t2;
                    if (r && t1.IsVector && t2.IsVector)
                    {
                        var v1 = t1.AsVector;
                        var v2 = t2.AsVector;
                        r = v1.DimCount != v2.DimCount || v1.KeyCount != v2.KeyCount;
                        r |= v1.RawKind != v2.RawKind;
                        r |= v1.ItemType != v2.ItemType;
                        r |= v1.IsKnownSizeVector != v2.IsKnownSizeVector;
                    }
                    if (r)
                    {
                        if (raise)
                        {
                            throw Contracts.Except("Column type {0} is different {1} != {2}\nS1: {3}\nS2: {4}",
                                    i, t1, t2, ToString(sch1), ToString(sch2));
                        }
                        else
                            return false;
                    }
                }
            }
            return true;
        }

        /// <summary>
        /// Saves a type into a stream.
        /// </summary>
        public static void WriteType(ModelSaveContext ctx, ColumnType type)
        {
            ctx.Writer.Write(type.IsVector);
            if (type.IsVector)
            {
                ctx.Writer.Write(type.AsVector.DimCount);
                for (int i = 0; i < type.AsVector.DimCount; ++i)
                    ctx.Writer.Write(type.AsVector.GetDim(i));
                ctx.Writer.Write((byte)type.AsVector.ItemType.RawKind);
            }
            else if (type.IsKey)
                throw Contracts.ExceptNotImpl("Key cannot be serialized yet.");
            else
                ctx.Writer.Write((byte)type.RawKind);
        }

        /// <summary>
        /// From DataKind to ColumnType
        /// </summary>
        public static ColumnType FromKind(DataKind kind)
        {
            switch (kind)
            {
                case DataKind.TX:
                    return TextType.Instance;
                case DataKind.Bool:
                    return BoolType.Instance;
                case DataKind.DateTime:
                    return DateTimeType.Instance;
                case DataKind.TimeSpan:
                    return TimeSpanType.Instance;
                default:
                    return NumberType.FromKind(kind);
            }
        }

        public static ColumnType ReadType(ModelLoadContext ctx)
        {
            bool isVector = ctx.Reader.ReadBoolean();
            if (isVector)
            {
                int dimCount = ctx.Reader.ReadInt32();
                if (dimCount != 1)
                    throw Contracts.ExceptNotImpl("Number of dimensions should be 1.");
                var dims = new int[dimCount];
                for (int i = 0; i < dimCount; ++i)
                    dims[i] = ctx.Reader.ReadInt32();
                var kind = (DataKind)ctx.Reader.ReadByte();
                return new VectorType(PrimitiveType.FromKind(kind), dims[0]);
            }
            else
            {
                var kind = (DataKind)ctx.Reader.ReadByte();
                return FromKind(kind);
            }
        }

        /// <summary>
        /// To avoid code duplication in many transform which convert a column into another one.
        /// </summary>
        public class OneToOneColumnForArgument : OneToOneColumn
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The result type", ShortName = "type")]
            public DataKind? ResultType;

            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key")]
            public KeyRange KeyRange;

            public static OneToOneColumnForArgument Parse(string str)
            {
                var res = new OneToOneColumnForArgument();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            protected override bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // We accept N:T:S where N is the new column name, T is the new type,
                // and S is source column names.
                string extra;
                if (!base.TryParse(str, out extra))
                    return false;
                if (extra == null)
                    return true;

                DataKind kind;
                if (!TypeParsingUtils.TryParseDataKind(extra, out kind, out KeyRange))
                    return false;
                ResultType = kind == default(DataKind) ? default(DataKind?) : kind;
                return true;
            }

            public override string ToString()
            {
                var sb = new StringBuilder();
                if (!TryUnparse(sb))
                    throw Contracts.Except("Unable to convert a column into string.");
                return sb.ToString();
            }

            public bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                if (ResultType == null && KeyRange == null)
                    return TryUnparseCore(sb);

                if (!TrySanitize())
                    return false;
                if (CmdQuoter.NeedsQuoting(Name) || CmdQuoter.NeedsQuoting(Source))
                    return false;

                int ich = sb.Length;
                sb.Append(Name);
                sb.Append(':');
                if (ResultType != null)
                    sb.Append(ResultType.Value.GetString());
                if (KeyRange != null)
                {
                    sb.Append('[');
                    if (!KeyRange.TryUnparse(sb))
                    {
                        sb.Length = ich;
                        return false;
                    }
                    sb.Append(']');
                }
                sb.Append(':');
                sb.Append(Source);
                return true;
            }
        }

        /// <summary>
        /// Suggests a default schema for a predictor
        /// </summary>
        /// <param name="kind"></param>
        /// <param name="dim"></param>
        /// <returns></returns>
        public static ISchema PredictionDefaultSchema(PredictionKind kind, int dim = 0)
        {
            switch (kind)
            {
                case PredictionKind.BinaryClassification:
                    return new ExtendedSchema(null, new[] { "Score" }, new[] { NumberType.R4 });
                case PredictionKind.Regression:
                    return new ExtendedSchema(null, new[] { "Prediction" }, new[] { NumberType.R4 });
                case PredictionKind.MultiClassClassification:
                    return new ExtendedSchema(null, new[] { "Scores" }, new[] { new VectorType(NumberType.R4, dim) });
                case PredictionKind.MultiOutputRegression:
                    return new ExtendedSchema(null, new[] { "Predictions" }, new[] { new VectorType(NumberType.R4, dim) });
                default:
                    throw Contracts.Except("Unable to build the schema for kind {0}", kind);
            }
        }

        public static Data.DataKind DataKind2DataDataKind(DataKind kind)
        {
            return (Data.DataKind)kind;
        }

        public static Data.TextLoaderColumn[] ToColumnArgArray(ISchema schema)
        {
            var res = new Data.TextLoaderColumn[schema.ColumnCount];
            for (int i = 0; i < res.Length; ++i)
            {
                res[i] = new Data.TextLoaderColumn()
                {
                    Name = schema.GetColumnName(i),
                    Type = DataKind2DataDataKind(schema.GetColumnType(i).RawKind),
                    Source = new[] { new Data.TextLoaderRange(i) }
                };
            }
            return res;
        }

        /// <summary>
        /// Returns the data kind based on a type.
        /// </summary>
        public static DataKind GetKind<TLabel>()
        {
            if (typeof(TLabel) == typeof(DvBool) || typeof(TLabel) == typeof(bool))
                return DataKind.BL;
            if (typeof(TLabel) == typeof(byte))
                return DataKind.U1;
            if (typeof(TLabel) == typeof(ushort))
                return DataKind.U2;
            if (typeof(TLabel) == typeof(uint))
                return DataKind.U4;
            if (typeof(TLabel) == typeof(int) || typeof(TLabel) == typeof(DvInt4))
                return DataKind.I4;
            if (typeof(TLabel) == typeof(Int64) || typeof(TLabel) == typeof(DvInt8))
                return DataKind.I8;
            if (typeof(TLabel) == typeof(float))
                return DataKind.R4;
            if (typeof(TLabel) == typeof(double))
                return DataKind.R8;
            if (typeof(TLabel) == typeof(DvText) || typeof(TLabel) == typeof(string))
                return DataKind.TX;
            throw Contracts.ExceptNotSupp("Unsupported output type {0}.", typeof(TLabel));
        }

        public static ColumnType GetColumnType<TLabel>()
        {
            var kind = GetKind<TLabel>();
            switch (kind)
            {
                case DataKind.BL:
                    return BoolType.Instance;
                case DataKind.U1:
                    return NumberType.U1;
                case DataKind.U2:
                    return NumberType.U2;
                case DataKind.U4:
                    return NumberType.U4;
                case DataKind.I4:
                    return NumberType.I4;
                case DataKind.R4:
                    return NumberType.R4;
                default:
                    throw Contracts.ExceptNotSupp($"Unsupported output type: {kind}.");
            }
        }

        public static ValueMapper<TLabel, TDest> GetConverter<TLabel, TDest>(out bool identity)
        {
            var col1 = GetColumnType<TLabel>();
            var col2 = GetColumnType<TDest>();
            if (typeof(TLabel) == typeof(float))
            {
                if (typeof(TDest) == typeof(uint))
                {
                    ValueMapper<float, uint> temp = (ref float src, ref uint dst) =>
                    {
                        if (src < 0)
                            throw Contracts.ExceptValue("Unable to converter {0} '{1}' into {2}.", typeof(float).ToString(), src, typeof(uint));
                        dst = (uint)src;
                    };
                    identity = false;
                    return temp as ValueMapper<TLabel, TDest>;
                }
                else if (typeof(TDest) == typeof(int))
                {
                    ValueMapper<float, int> temp = (ref float src, ref int dst) =>
                    {
                        dst = (int)src;
                    };
                    identity = false;
                    return temp as ValueMapper<TLabel, TDest>;
                }
            }
            return Conversions.Instance.GetStandardConversion<TLabel, TDest>(col1, col2, out identity);
        }
    }
}
