using System.Collections.ObjectModel;
using System.Diagnostics;

namespace LandauDistributionFP64.InternalUtils {
    internal static class ApproxUtil {

        public static double Pade(double x, (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) table) {
            double sc = Poly(x, table.numer), sd = Poly(x, table.denom);

            Debug.Assert(sd >= 0.5, $"pade denom digits loss! {x}");

            return sc / sd;
        }

        public static double Poly(double x, ReadOnlyCollection<double> table) {
            double s = table[^1];

            for (int i = table.Count - 2; i >= 0; i--) {
                s = s * x + table[i];
            }

            return s;
        }
    }
}