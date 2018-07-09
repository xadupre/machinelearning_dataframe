// See the LICENSE file in the project root for more information.

using System;


namespace Microsoft.ML.Ext.DataManipulation
{
    public interface ITUple
    {
        int Length { get; }
    }

    /// <summary>
    /// Tuple but every item can be modified.
    /// </summary>
    public class MutableTuple<T1> : IEquatable<MutableTuple<T1>>, IComparable<MutableTuple<T1>>, ITUple
        where T1 : IEquatable<T1>, IComparable<T1>
    {
        public T1 Item1;
        public int Length => 1;

        public MutableTuple() { }
        public bool Equals(MutableTuple<T1> value) { return Item1.Equals(value.Item1); }
        public int CompareTo(MutableTuple<T1> value) { return Item1.CompareTo(value.Item1); }
        public Tuple<T1> ToTuple() { return new Tuple<T1>(Item1); }
        public ImmutableTuple<T1> ToImTuple() { return new ImmutableTuple<T1>(Item1); }
    }

    /// <summary>
    /// Comparable tuple.
    /// </summary>
    public class ImmutableTuple<T1> : Tuple<T1>, IEquatable<ImmutableTuple<T1>>, IComparable<ImmutableTuple<T1>>, ITUple
        where T1 : IEquatable<T1>, IComparable<T1>
    {
        public int Length => 1;
        public ImmutableTuple(T1 t1) : base(t1) { }
        public bool Equals(ImmutableTuple<T1> value) { return Item1.Equals(value.Item1); }
        public int CompareTo(ImmutableTuple<T1> value) { return Item1.CompareTo(value.Item1); }
    }

    /// <summary>
    /// Tuple but every item can be modified.
    /// </summary>
    public class MutableTuple<T1, T2> : IEquatable<MutableTuple<T1, T2>>, IComparable<MutableTuple<T1, T2>>, ITUple
        where T1 : IEquatable<T1>, IComparable<T1>
        where T2 : IEquatable<T2>, IComparable<T2>
    {
        public T1 Item1;
        public T2 Item2;
        public int Length => 2;

        public MutableTuple() { }
        public bool Equals(MutableTuple<T1, T2> value) { return Item1.Equals(value.Item1) && Item2.Equals(value.Item2); }
        public int CompareTo(MutableTuple<T1, T2> value)
        {
            int r = Item1.CompareTo(value.Item1);
            return r == 0 ? Item2.CompareTo(value.Item2) : r;
        }
        public Tuple<T1, T2> ToTuple() { return new Tuple<T1, T2>(Item1, Item2); }
        public ImmutableTuple<T1, T2> ToImTuple() { return new ImmutableTuple<T1, T2>(Item1, Item2); }
    }

    /// <summary>
    /// Comparable tuple.
    /// </summary>
    public class ImmutableTuple<T1, T2> : Tuple<T1, T2>, IEquatable<ImmutableTuple<T1, T2>>, IComparable<ImmutableTuple<T1, T2>>, ITUple
        where T1 : IEquatable<T1>, IComparable<T1>
        where T2 : IEquatable<T2>, IComparable<T2>
    {
        public int Length => 2;
        public ImmutableTuple(T1 t1, T2 t2) : base(t1, t2) { }
        public bool Equals(ImmutableTuple<T1, T2> value) { return Item1.Equals(value.Item1) && Item2.Equals(value.Item2); }
        public int CompareTo(ImmutableTuple<T1, T2> value)
        {
            int r = Item1.CompareTo(value.Item1);
            return r == 0 ? Item2.CompareTo(value.Item2) : r;
        }
    }

    /// <summary>
    /// Tuple but every item can be modified.
    /// </summary>
    public class MutableTuple<T1, T2, T3> : IEquatable<MutableTuple<T1, T2, T3>>, IComparable<MutableTuple<T1, T2, T3>>, ITUple
        where T1 : IEquatable<T1>, IComparable<T1>
        where T2 : IEquatable<T2>, IComparable<T2>
        where T3 : IEquatable<T3>, IComparable<T3>
    {
        public T1 Item1;
        public T2 Item2;
        public T3 Item3;
        public int Length => 3;

        public MutableTuple() { }
        public bool Equals(MutableTuple<T1, T2, T3> value)
        {
            return Item1.Equals(value.Item1) && Item2.Equals(value.Item2) && Item3.Equals(value.Item3);
        }
        public int CompareTo(MutableTuple<T1, T2, T3> value)
        {
            int r = Item1.CompareTo(value.Item1);
            if (r != 0)
                return r;
            r = Item2.CompareTo(value.Item2);
            if (r != 0)
                return r;
            return Item3.CompareTo(value.Item3);
        }
        public Tuple<T1, T2, T3> ToTuple() { return new Tuple<T1, T2, T3>(Item1, Item2, Item3); }
        public ImmutableTuple<T1, T2, T3> ToImTuple() { return new ImmutableTuple<T1, T2, T3>(Item1, Item2, Item3); }
    }

    /// <summary>
    /// Comparable tuple.
    /// </summary>
    public class ImmutableTuple<T1, T2, T3> : Tuple<T1, T2, T3>, IEquatable<ImmutableTuple<T1, T2, T3>>, IComparable<ImmutableTuple<T1, T2, T3>>, ITUple
        where T1 : IEquatable<T1>, IComparable<T1>
        where T2 : IEquatable<T2>, IComparable<T2>
        where T3 : IEquatable<T3>, IComparable<T3>
    {
        public int Length => 3;
        public ImmutableTuple(T1 t1, T2 t2, T3 t3) : base(t1, t2, t3) { }
        public bool Equals(ImmutableTuple<T1, T2, T3> value)
        {
            return Item1.Equals(value.Item1) && Item2.Equals(value.Item2) && Item3.Equals(value.Item3);
        }
        public int CompareTo(ImmutableTuple<T1, T2, T3> value)
        {
            int r = Item1.CompareTo(value.Item1);
            if (r != 0)
                return r;
            r = Item2.CompareTo(value.Item2);
            if (r != 0)
                return r;
            return Item3.CompareTo(value.Item3);
        }
    }
}
