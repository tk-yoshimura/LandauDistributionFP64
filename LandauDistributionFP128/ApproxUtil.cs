using MultiPrecision;
using System.Collections.ObjectModel;
using System.Diagnostics;

namespace LandauDistributionFP128.InternalUtils {
    internal static class ApproxUtil {

        public static MultiPrecision<Pow2.N4> Pade(MultiPrecision<Pow2.N4> x, (ReadOnlyCollection<MultiPrecision<Pow2.N4>> numer, ReadOnlyCollection<MultiPrecision<Pow2.N4>> denom) table) {
            MultiPrecision<Pow2.N4> sc = Poly(x, table.numer), sd = Poly(x, table.denom);

            Debug.Assert(sd >= 0.5, $"pade denom digits loss! {x}");

            return sc / sd;
        }

        public static MultiPrecision<Pow2.N4> Poly(MultiPrecision<Pow2.N4> x, ReadOnlyCollection<MultiPrecision<Pow2.N4>> table) {
            MultiPrecision<Pow2.N4> s = table[^1];

            for (int i = table.Count - 2; i >= 0; i--) {
                s = s * x + table[i];
            }

            return s;
        }
    }
}