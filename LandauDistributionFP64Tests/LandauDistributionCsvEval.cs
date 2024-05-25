using DoubleDouble;
using System.Diagnostics;

namespace LandauDistributionFP64Tests {
    [TestClass()]
    public class LandauDistributionCsvEval {

        [TestMethod()]
        public void PlotPDF() {
            using StreamReader sr = new("../../../../results/landau_pdf_cpp.csv");
            sr.ReadLine();

            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            ddouble max_rateerror = 0;

            for (double x = -6; x <= 64; x += 1d / 1024) {
                ddouble actual = sr.ReadLine().Split(',')[1];
                ddouble expected = dist_fp128.PDF(x);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                max_rateerror = ddouble.Max(max_rateerror, rateerror);

                Debug.WriteLine($"{x},{rateerror:e8}");
            }

            Console.WriteLine($"{nameof(max_rateerror)}: {max_rateerror:e8}");
        }

        [TestMethod()]
        public void PlotPDFLimit() {
            using StreamReader sr = new("../../../../results/landau_pdf_limit_cpp.csv");
            sr.ReadLine();

            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            ddouble max_rateerror = 0;

            for (double x0 = 64; x0 <= double.ScaleB(1, 64); x0 *= 2) {
                for (double x = x0; x < x0 * 2; x += x0 / 256) {
                    ddouble actual = sr.ReadLine().Split(',')[1];
                    ddouble expected = dist_fp128.PDF(x);
                    ddouble error = ddouble.Abs(expected - actual);
                    ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                    max_rateerror = ddouble.Max(max_rateerror, rateerror);

                    Debug.WriteLine($"{x},{rateerror:e8}");
                }
            }

            Console.WriteLine($"{nameof(max_rateerror)}: {max_rateerror:e8}");
        }

        [TestMethod()]
        public void PlotCDFLower() {
            using StreamReader sr = new("../../../../results/landau_cdf_cpp.csv");
            sr.ReadLine();

            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            ddouble max_rateerror = 0;

            for (double x = -6; x <= 64; x += 1d / 1024) {
                ddouble actual = sr.ReadLine().Split(',')[1];
                ddouble expected = dist_fp128.CDF(x, DoubleDoubleStatistic.Interval.Lower);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                max_rateerror = ddouble.Max(max_rateerror, rateerror);

                Debug.WriteLine($"{x},{rateerror:e8}");
            }

            Console.WriteLine($"{nameof(max_rateerror)}: {max_rateerror:e8}");
        }

        [TestMethod()]
        public void PlotCDFUpperLimit() {
            using StreamReader sr = new("../../../../results/landau_cdf_limit_cpp.csv");
            sr.ReadLine();

            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            ddouble max_rateerror = 0;

            for (double x0 = 64; x0 <= double.ScaleB(1, 64); x0 *= 2) {
                for (double x = x0; x < x0 * 2; x += x0 / 256) {
                    ddouble actual = sr.ReadLine().Split(',')[1];
                    ddouble expected = dist_fp128.CDF(x, DoubleDoubleStatistic.Interval.Upper);
                    ddouble error = ddouble.Abs(expected - actual);
                    ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                    max_rateerror = ddouble.Max(max_rateerror, rateerror);

                    Debug.WriteLine($"{x},{rateerror:e8}");
                }
            }

            Console.WriteLine($"{nameof(max_rateerror)}: {max_rateerror:e8}");
        }

        [TestMethod()]
        public void PlotQuantile() {
            using StreamReader sr = new("../../../../results/landau_quantile_cpp.csv");
            sr.ReadLine();

            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            ddouble max_rateerror = 0;

            for (double x = 1d / 8192; x < 1; x += 1d / 8192) {
                ddouble actual = sr.ReadLine().Split(',')[1];
                ddouble expected = dist_fp128.Quantile(x, DoubleDoubleStatistic.Interval.Lower);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                max_rateerror = ddouble.Max(max_rateerror, rateerror);

                Debug.WriteLine($"{x},{rateerror:e8}");
            }

            Console.WriteLine($"{nameof(max_rateerror)}: {max_rateerror:e8}");
        }

        [TestMethod()]
        public void PlotQuantileLower() {
            using StreamReader sr = new("../../../../results/landau_quantilelower_limit_cpp.csv");
            sr.ReadLine();

            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            ddouble max_rateerror = 0;

            for (double x = 1d / 8192; x > double.ScaleB(1, -1000); x /= 2) {
                ddouble actual = sr.ReadLine().Split(',')[1];
                ddouble expected = dist_fp128.Quantile(x, DoubleDoubleStatistic.Interval.Lower);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                max_rateerror = ddouble.Max(max_rateerror, rateerror);

                Debug.WriteLine($"{x},{rateerror:e8}");
            }

            Console.WriteLine($"{nameof(max_rateerror)}: {max_rateerror:e8}");
        }

        [TestMethod()]
        public void PlotQuantileUpper() {
            using StreamReader sr = new("../../../../results/landau_quantileupper_limit_cpp.csv");
            sr.ReadLine();

            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            ddouble max_rateerror = 0;

            for (double x0 = 1d / 8192; x0 > double.ScaleB(1, -128); x0 /= 2) {
                for (double x = x0; x > x0 / 2; x -= x0 / 256) {
                    ddouble actual = sr.ReadLine().Split(',')[1];
                    ddouble expected = dist_fp128.Quantile(x, DoubleDoubleStatistic.Interval.Upper);
                    ddouble error = ddouble.Abs(expected - actual);
                    ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                    max_rateerror = ddouble.Max(max_rateerror, rateerror);

                    Debug.WriteLine($"{x},{rateerror:e8}");
                }
            }

            Console.WriteLine($"{nameof(max_rateerror)}: {max_rateerror:e8}");
        }
    }
}