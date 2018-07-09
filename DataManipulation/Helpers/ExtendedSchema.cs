// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Extends an existing Schema.
    /// </summary>
    public class ExtendedSchema : ISchema
    {
        readonly ISchema _schemaInput;
        readonly string[] _names;
        readonly ColumnType[] _types;
        readonly Dictionary<string, int> _maprev;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="inputSchema">existing schema</param>
        /// <param name="names">new columns</param>
        /// <param name="types">corresponding types</param>
        public ExtendedSchema(ISchema inputSchema, string[] names, ColumnType[] types)
        {
            _schemaInput = inputSchema;
            if (names == null || names.Length == 0)
                throw Contracts.ExceptEmpty("The extended schema must contain new names.");
            if (types == null || types.Length != names.Length)
                throw Contracts.Except("names and types must have the same length.");
            _names = names;
            _types = types;
            _maprev = new Dictionary<string, int>();
            for (int i = 0; i < _names.Length; ++i)
            {
                if (_maprev.ContainsKey(_names[i]))
                    throw Contracts.Except("Column '{0}' was added twice. This is not allowed.", _names[i]);
                _maprev[_names[i]] = i;
            }
        }

        /// <summary>
        /// Concatenation of two schemas.
        /// </summary>
        public static ExtendedSchema operator +(ExtendedSchema sch1, ExtendedSchema sch2)
        {
            if (sch1._schemaInput != sch2._schemaInput)
                throw new Exception("ExtendedSchema can be merged if theyy share the same input schema.");
            var names = new List<string>();
            var types = new List<ColumnType>();
            names.AddRange(sch1._names);
            names.AddRange(sch2._names);
            types.AddRange(sch1._types);
            types.AddRange(sch2._types);
            return new ExtendedSchema(sch1._schemaInput, names.ToArray(), types.ToArray());
        }

        /// <summary>
        /// Parses a schema and creates an ExtendedSchema for it.
        /// </summary>
        public static ExtendedSchema Parse(string schema, IChannel ch = null)
        {
            var cols = schema.Split(new char[] { ' ', ';' }, StringSplitOptions.RemoveEmptyEntries);
            var tlcols = new List<TextLoader.Column>();
            for (int i = 0; i < cols.Length; ++i)
            {
                if (cols[i].StartsWith("col="))
                    cols[i] = cols[i].Substring(4);
                if (cols[i].Contains(":Vec<"))
                {
                    var dot = cols[i].Split(':');
                    if (dot.Length != 3)
                        throw ch != null ? ch.Except("Expects 3 parts in '{0}'", cols[i])
                                         : Contracts.Except("Expects 3 parts in '{0}'", cols[i]);
                    if (!dot[1].StartsWith("Vec<"))
                        throw ch != null ? ch.Except("Unable to parse '{0}'", cols[i])
                                         : Contracts.Except("Unable to parse '{0}'", cols[i]);
                    if (!dot[1].EndsWith(">"))
                        throw ch != null ? ch.Except("Unable to parse '{0}'", cols[i])
                                         : Contracts.Except("Unable to parse '{0}'", cols[i]);
                    var temp = dot[1].Substring(4);
                    temp = temp.Substring(0, temp.Length - 1);
                    var splc = temp.Split(',');
                    if (splc.Length != 2)
                        throw ch != null ? ch.Except("Unable to parse '{0}'", cols[i])
                                         : Contracts.Except("Unable to parse '{0}'", cols[i]);
                    int last, nb;
                    try
                    {
                        last = int.Parse(dot[2]);
                        nb = int.Parse(splc[1]);
                    }
                    catch
                    {
                        throw ch != null ? ch.Except("Unable to parse '{0}'", cols[i])
                                         : Contracts.Except("Unable to parse '{0}'", cols[i]);
                    }
                    dot[1] = splc[0];
                    dot[2] = string.Format("{0}-{1}", last, last + nb - 1);
                    cols[i] = string.Format("{0}:{1}:{2}", dot[0], dot[1], dot[2]);
                }
                else if (cols[i].Contains(":Key<"))
                {
                    var dot = cols[i].Split(':');
                    if (dot.Length == 4)
                    {
                        // GroupId:Key<U4, Min:0>:8
                        dot = new[] { dot[0], dot[1] + ":" + dot[2], dot[3] };
                        if (dot[1].Contains(",Min:0>"))
                            dot[1] = dot[1].Replace(",Min:0>", ",0-*>");
                        else
                            throw ch != null ? ch.Except("Expects 3 parts in '{0}'", cols[i])
                                             : Contracts.Except("Expects 3 parts in '{0}'", cols[i]);
                    }
                    if (dot.Length != 3)
                        throw ch != null ? ch.Except("Expects 3 parts in '{0}'", cols[i])
                                         : Contracts.Except("Expects 3 parts in '{0}'", cols[i]);
                    if (!dot[1].StartsWith("Key<"))
                        throw ch != null ? ch.Except("Unable to parse '{0}'", cols[i])
                                         : Contracts.Except("Unable to parse '{0}'", cols[i]);
                    if (!dot[1].EndsWith(">"))
                        throw ch != null ? ch.Except("Unable to parse '{0}'", cols[i])
                                         : Contracts.Except("Unable to parse '{0}'", cols[i]);
                    var temp = dot[1].Substring(4);
                    temp = temp.Substring(0, temp.Length - 1);
                    var splc = temp.Split(',');
                    if (splc.Length != 2)
                        throw ch != null ? ch.Except("Unable to parse '{0}'", cols[i])
                                         : Contracts.Except("Unable to parse '{0}'", cols[i]);
                    dot[1] = string.Format("{0}[{1}]", splc[0], splc[1]);
                    cols[i] = string.Format("{0}:{1}:{2}", dot[0], dot[1], dot[2]);
                }
                var t = TextLoader.Column.Parse(cols[i]);
                if (t == null)
                    t = TextLoader.Column.Parse(string.Format("{0}:{1}", cols[i], i));
                if (t == null)
                    throw ch != null ? ch.Except("Unable to parse '{0}' or '{1}'", cols[i], string.Format("{0}:{1}", cols[i], i))
                                     : Contracts.Except("Unable to parse '{0}' or '{1}'", cols[i], string.Format("{0}:{1}", cols[i], i));
                tlcols.Add(t);
            }
            return new ExtendedSchema(null, tlcols.Select(c => c.Name).ToArray(), tlcols.Select(c => SchemaHelper.Convert(c, ch)).ToArray());
        }

        /// <summary>
        /// Returns the extended number of columns.
        /// </summary>
        public int ColumnCount { get { return (_schemaInput == null ? 0 : _schemaInput.ColumnCount) + _names.Length; } }

        /// <summary>
        /// Returns the index of a column. If multiple columns
        /// share the same name, it returns the last one.
        /// </summary>
        public int GetColumnIndex(string name)
        {
            int res;
            var r = TryGetColumnIndex(name, out res);
            if (r)
                return res;
            throw new IndexOutOfRangeException(string.Format("Unable to find column '{0}'.", name));
        }

        /// <summary>
        /// If multiple columns share the same name, the function tries to find
        /// a name in the new columns and then in the previous schema.
        /// That follows the specifications: if two columns
        /// share the same name, the first is considered as hidden.
        /// </summary>
        public bool TryGetColumnIndex(string name, out int res)
        {
            if (string.IsNullOrEmpty(name))
                throw new IndexOutOfRangeException("Unable to find empty column.");
            int nb = _schemaInput == null ? 0 : _schemaInput.ColumnCount;

            if (_maprev.ContainsKey(name))
            {
                res = _maprev[name] + nb;
                return true;
            }

            if (_schemaInput != null)
            {
                var r = _schemaInput.TryGetColumnIndex(name, out res);
                if (r)
                    return true;
            }

            res = -1;
            return false;
        }

        /// <summary>
        /// Returns the column name for column <i>col</i>.
        /// </summary>
        public string GetColumnName(int col)
        {
            int count = _schemaInput == null ? 0 : _schemaInput.ColumnCount;
            if (col < count)
                return _schemaInput.GetColumnName(col);
            if (col < ColumnCount)
                return _names[col - count];
            throw new IndexOutOfRangeException();
        }

        /// <summary>
        /// Returns the column type for column <i>col</i>.
        /// </summary>
        public ColumnType GetColumnType(int col)
        {
            int count;
            if (_schemaInput != null)
            {
                count = _schemaInput.ColumnCount;
                if (col < count)
                    return _schemaInput.GetColumnType(col);
            }
            else
                count = 0;
            if (col < ColumnCount)
                return _types[col - count];
            throw new IndexOutOfRangeException();
        }

        /// <summary>
        /// Returns the metadata.
        /// </summary>
        public void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            int count = _schemaInput == null ? 0 : _schemaInput.ColumnCount;
            if (col < count)
            {
                _schemaInput.GetMetadata(kind, col, ref value);
                return;
            }
            if (kind == MetadataUtils.Kinds.SlotNames)
            {
                var res = GetSlotNames(col);
                DvText[] dres = new DvText[res.Length];
                for (int i = 0; i < res.Length; ++i)
                    dres[i] = new DvText(res[i]);
                var vec = new VBuffer<DvText>(res.Length, dres);
                ValueGetter<VBuffer<DvText>> conv = (ref VBuffer<DvText> val) => { val = vec; };
                var conv2 = conv as ValueGetter<TValue>;
                conv2(ref value);
                return;
            }

            int index;
            if (TryGetColumnIndex(kind, out index))
            {
                if (typeof(TValue) == typeof(DvText))
                {
                    ValueMapper<string, DvText> convs = (ref string src, ref DvText dst) =>
                    {
                        dst = new DvText(src);
                    };
                    var convs2 = convs as ValueMapper<string, TValue>;
                    convs2(ref kind, ref value);
                }
            }
            else
                throw new IndexOutOfRangeException();
        }

        /// <summary>
        /// Returns all slots names for column <i>col</i>.
        /// </summary>
        string[] GetSlotNames(int col)
        {
            string name = GetColumnName(col);
            var type = GetColumnType(col);
            if (type.IsVector)
            {
                var vec = type.AsVector;
                if (vec.DimCount != 1)
                    throw Contracts.ExceptNotImpl("Only one dimension is implemented.");
                var res = new string[vec.GetDim(0)];
                for (int i = 0; i < res.Length; ++i)
                    res[i] = string.Format("{0}{1}", name, i);
                return res;
            }
            else
                return new string[] { name };
        }

        /// <summary>
        /// Returns the metadata.
        /// </summary>
        public ColumnType GetMetadataTypeOrNull(string kind, int col)
        {
            int count = _schemaInput == null ? 0 : _schemaInput.ColumnCount;
            if (col < count)
                return _schemaInput.GetMetadataTypeOrNull(kind, col);
            if (kind == MetadataUtils.Kinds.SlotNames)
            {
                var ty = GetColumnType(col);
                if (ty.IsVector && ty.AsVector.DimCount == 1 && ty.AsVector.GetDim(0) > 0)
                    return new VectorType(TextType.Instance, ty.AsVector.GetDim(0));
            }
            return null;
        }

        /// <summary>
        /// Returns an enumerator on the metadata.
        /// </summary>
        public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
        {
            int count = _schemaInput == null ? 0 : _schemaInput.ColumnCount;
            if (col < count)
            {
                foreach (var s in _schemaInput.GetMetadataTypes(col))
                    yield return s;
            }
            else if (col < ColumnCount)
            {
                int c = col - _schemaInput.ColumnCount;
                yield return new KeyValuePair<string, ColumnType>(_names[c], _types[c]);
            }
            else
                throw new IndexOutOfRangeException();
        }
    }
}