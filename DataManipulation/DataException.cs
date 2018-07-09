// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Ext.DataManipulation
{
    /// <summary>
    /// Raised when there is a type mismatch.
    /// </summary>
    public class DataTypeError : Exception
    {
        public DataTypeError(string msg) : base(msg)
        {
        }
    }

    /// <summary>
    /// Raised with irreconceliable values.
    /// </summary>
    public class DataValueError : Exception
    {
        public DataValueError(string msg) : base(msg)
        {
        }
    }

    /// <summary>
    /// Raised when there is an error in names.
    /// </summary>
    public class DataNameError : Exception
    {
        public DataNameError(string msg) : base(msg)
        {
        }
    }

    public class DataFrameViewException : Exception
    {
        /// <summary>
        /// Raised with a operator cannot be done with a DataFrameView.
        /// </summary>
        public DataFrameViewException(string msg) : base(msg)
        {
        }
    }
}
