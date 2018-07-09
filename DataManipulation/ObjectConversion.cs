// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;


namespace Microsoft.ML.Ext.DataManipulation
{
    public static class ObjectConversion
    {
        public static void Convert<T>(ref object src, out T value)
        {
            if ((src as string) != null)
            {
                var dv = new DvText((string)src);
                value = (T)(object)dv;
            }
            else
                value = (T)src;
        }
    }
}
