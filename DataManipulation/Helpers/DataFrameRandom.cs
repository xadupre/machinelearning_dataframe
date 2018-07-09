// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;


namespace Scikit.ML.DataFrame
{
    /// <summary>
    /// Implements grouping functions for dataframe.
    /// </summary>
    public static class DataFrameRandom
    {
        /// <summary>
        /// Draws n random integers in [0, N-1].
        /// They can be distinct or not.
        /// The function is not efficient if n is close to N and distinct is true.
        /// </summary>
        public static int[] RandomIntegers(int n, int N, bool distinct = false, IRandom rand = null)
        {
            var res = new int[n];
            if (rand == null)
                rand = new SysRandom();
            if (distinct)
            {
                if (n > N)
                    throw new DataValueError($"Cannot draw more than {N} distinct values.");
                var hash = new HashSet<int>();
                int nb = 0;
                int i;
                while (nb < n)
                {
                    i = rand.Next(N);
                    if (hash.Contains(i))
                        continue;
                    hash.Add(i);
                    res[nb] = i;
                    ++nb;
                }
            }
            else
            {
                for (int i = 0; i < n; ++i)
                    res[i] = rand.Next(N);
            }
            return res;
        }
    }
}
