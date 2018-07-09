// See the LICENSE file in the project root for more information.


using System.IO;
using System.Reflection;
using Microsoft.ML.Runtime.Tools;


namespace UnitTests
{
    public static class FileHelper
    {
        static bool IsRoot(string root)
        {
            if (!Directory.Exists(root))
                return false;
            var dotRoot = Path.Combine(root, "LICENSE");
            if (!File.Exists(dotRoot))
                return false;
            return true;
        }

        public static string GetRoot()
        {
            string currentDirectory = Directory.GetCurrentDirectory();
            currentDirectory = Path.GetFullPath(currentDirectory);
            var root = currentDirectory;
            while (!string.IsNullOrEmpty(root))
            {
                if (IsRoot(root))
                    return root;
                root = Path.GetDirectoryName(root);
            }
            throw new DirectoryNotFoundException(string.Format("Unable to find root folder from '{0}'", currentDirectory));
        }

        /// <summary>
        /// Returns the ML.net version.
        /// </summary>
        public static string GetUsedMLVersion()
        {
            return typeof(VersionCommand).GetTypeInfo().Assembly.GetName().Version.ToString();
        }

        /// <summary>
        /// Returns a data relative to folder data.
        /// </summary>
        /// <param name="name">name of the dataset</param>
        /// <returns>full path of the dataset</returns>
        public static string GetTestFile(string name)
        {
            var root = GetRoot();
            var full = Path.Combine(root, "data", name);
            if (!File.Exists(full))
                throw new FileNotFoundException(string.Format("Unable to find '{0}'\nFull='{1}'\nroot='{2}'\ncurrent='{3}'.",
                                    name, full, root, Path.GetFullPath(Directory.GetCurrentDirectory())));
            return full;
        }

        /// <summary>
        /// Creates a folder where the results of the unittests should be placed.
        /// </summary>
        /// <param name="name">name of the output</param>
        /// <param name="testFunction">name of the test</param>
        /// <param name="extended">addition to make to the file</param>
        /// <returns></returns>
        public static string GetOutputFile(string name, string testFunction, params string[] extended)
        {
#if (DEBUG)
            var version = "Debug";
#else
            var version = "Release";
#endif
            var vers = GetUsedMLVersion();
            var root = GetRoot();
            var tests = Path.Combine(root, "_tests");
            var unittest = Path.Combine(tests, vers, version, testFunction);

            if (!Directory.Exists(unittest))
                Directory.CreateDirectory(unittest);

            if (extended != null && extended.Length > 0)
            {
                var jpl = string.Join("-", extended);
                unittest = Path.Combine(unittest, jpl);
                if (!Directory.Exists(unittest))
                    Directory.CreateDirectory(unittest);
            }
            var full = Path.Combine(unittest, name);
            if (File.Exists(full))
                File.Delete(full);
            return full;
        }
    }
}
