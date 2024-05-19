using DoubleDouble;

namespace LandauDistributionFP64Tests {
    [TestClass()]
    public class LandauDistributionSummary {

        static LandauDistributionSummary() {
            Directory.CreateDirectory("../../../../results/");
        }

        [TestMethod()]
        public void PlotPDF() {
            using StreamWriter sw = new("../../../../results/pdf_approx.csv");

            sw.WriteLine("x,y_expected,y_actual,error(abs),error(rate)");

            LandauDistributionFP64.LandauDistribution dist_fp64 = new();
            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            for (double x = -6; x <= 64; x += 1d / 1024) {
                double actual = dist_fp64.PDF(x);
                ddouble expected = dist_fp128.PDF(x);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                sw.WriteLine($"{x},{expected:e20},{actual},{error:e8},{rateerror:e8}");
            }
        }

        [TestMethod()]
        public void PlotPDFLimit() {
            using StreamWriter sw = new("../../../../results/pdflimit_approx.csv");

            sw.WriteLine("x,y_expected,y_actual,error(abs),error(rate)");

            LandauDistributionFP64.LandauDistribution dist_fp64 = new();
            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            for (double x0 = 64; x0 <= double.ScaleB(1, 64); x0 *= 2) {
                for (double x = x0; x < x0 * 2; x += x0 / 256) {
                    double actual = dist_fp64.PDF(x);
                    ddouble expected = dist_fp128.PDF(x);
                    ddouble error = ddouble.Abs(expected - actual);
                    ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                    sw.WriteLine($"{x},{expected:e20},{actual},{error:e8},{rateerror:e8}");
                }
            }
        }

        [TestMethod()]
        public void PlotCDFLower() {
            using StreamWriter sw = new("../../../../results/cdflower_approx.csv");

            sw.WriteLine("x,y_expected,y_actual,error(abs),error(rate)");

            LandauDistributionFP64.LandauDistribution dist_fp64 = new();
            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            for (double x = -6; x <= 64; x += 1d / 1024) {
                double actual = dist_fp64.CDF(x, LandauDistributionFP64.Interval.Lower);
                ddouble expected = dist_fp128.CDF(x, DoubleDoubleStatistic.Interval.Lower);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                sw.WriteLine($"{x},{expected:e20},{actual},{error:e8},{rateerror:e8}");
            }
        }

        [TestMethod()]
        public void PlotCDFUpperLimit() {
            using StreamWriter sw = new("../../../../results/cdfupperlimit_approx.csv");

            sw.WriteLine("x,y_expected,y_actual,error(abs),error(rate)");

            LandauDistributionFP64.LandauDistribution dist_fp64 = new();
            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            for (double x0 = 64; x0 <= double.ScaleB(1, 64); x0 *= 2) {
                for (double x = x0; x < x0 * 2; x += x0 / 256) {
                    double actual = dist_fp64.CDF(x, LandauDistributionFP64.Interval.Upper);
                    ddouble expected = dist_fp128.CDF(x, DoubleDoubleStatistic.Interval.Upper);
                    ddouble error = ddouble.Abs(expected - actual);
                    ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                    sw.WriteLine($"{x},{expected:e20},{actual},{error:e8},{rateerror:e8}");
                }
            }
        }

        [TestMethod()]
        public void PlotQuantile() {
            using StreamWriter sw = new("../../../../results/quantile_approx.csv");

            sw.WriteLine("x,y_expected,y_actual,error(abs),error(rate)");

            LandauDistributionFP64.LandauDistribution dist_fp64 = new();
            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            for (double x = 1d / 8192; x < 1; x += 1d / 8192) {
                double actual = dist_fp64.Quantile(x, LandauDistributionFP64.Interval.Lower);
                ddouble expected = dist_fp128.Quantile(x, DoubleDoubleStatistic.Interval.Lower);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                sw.WriteLine($"{x},{expected:e20},{actual},{error:e8},{rateerror:e8}");
            }
        }

        [TestMethod()]
        public void PlotQuantileNZ() {
            using StreamWriter sw = new("../../../../results/quantilenz_approx.csv");

            sw.WriteLine("x,y_expected,y_actual,error(abs),error(rate)");

            LandauDistributionFP64.LandauDistribution dist_fp64 = new();
            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            for (double x = 2457d / 8192; x < 3276d / 8192; x += 1d / 8192) {
                double actual = dist_fp64.Quantile(x, LandauDistributionFP64.Interval.Lower);
                ddouble expected = dist_fp128.Quantile(x, DoubleDoubleStatistic.Interval.Lower);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                sw.WriteLine($"{x},{expected:e20},{actual},{error:e8},{rateerror:e8}");
            }
        }

        [TestMethod()]
        public void PlotQuantileLower() {
            using StreamWriter sw = new("../../../../results/quantilelowerlimit_approx.csv");

            sw.WriteLine("x,y_expected,y_actual,error(abs),error(rate)");

            LandauDistributionFP64.LandauDistribution dist_fp64 = new();
            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            for (double x = 1d / 8192; x > double.ScaleB(1, -1000); x /= 2) {
                double actual = dist_fp64.Quantile(x, LandauDistributionFP64.Interval.Lower);
                ddouble expected = dist_fp128.Quantile(x, DoubleDoubleStatistic.Interval.Lower);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                sw.WriteLine($"{x},{expected:e20},{actual},{error:e8},{rateerror:e8}");
            }
        }

        [TestMethod()]
        public void PlotQuantileUpper() {
            using StreamWriter sw = new("../../../../results/quantileupperlimit_approx.csv");

            sw.WriteLine("x,y_expected,y_actual,error(abs),error(rate)");

            LandauDistributionFP64.LandauDistribution dist_fp64 = new();
            DoubleDoubleStatistic.ContinuousDistributions.LandauDistribution dist_fp128 = new();

            for (double x = 1d / 8192; x > double.ScaleB(1, -1000); x /= 2) {
                double actual = dist_fp64.Quantile(x, LandauDistributionFP64.Interval.Upper);
                ddouble expected = dist_fp128.Quantile(x, DoubleDoubleStatistic.Interval.Upper);
                ddouble error = ddouble.Abs(expected - actual);
                ddouble rateerror = (error != 0) ? error / ddouble.Abs(expected) : 0;

                sw.WriteLine($"{x},{expected:e20},{actual},{error:e8},{rateerror:e8}");
            }
        }
    }
}