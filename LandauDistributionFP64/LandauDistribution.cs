// Author and Approximation Formula Coefficient Generator: T.Yoshimura
// Github: https://github.com/tk-yoshimura
// Original Code: https://github.com/tk-yoshimura/LandauDistributionFP64

using LandauDistributionFP64.InternalUtils;
using LandauDistributionFP64.RandomGeneration;
using System.Collections.ObjectModel;
using System.Diagnostics;
using static System.Double;

namespace LandauDistributionFP64 {
    [DebuggerDisplay("{ToString(),nq}")]
    public class LandauDistribution {

        public double Mu { get; }

        public double C { get; }

        private readonly double c_inv, bias;

        private static readonly double mode_base = -0.42931452986133525017;
        private static readonly double median_base = 0.57563014394507821440;
        private static readonly double entropy_base = 2.37263644000448182448;

        public LandauDistribution() : this(mu: 0d, c: 1d) { }

        public LandauDistribution(double c) : this(mu: 0d, c: c) { }

        public LandauDistribution(double mu, double c) {
            if (!IsFinite(mu)) {
                throw new ArgumentOutOfRangeException(nameof(mu), "Invalid location parameter.");
            }
            if (!(c > 0 && IsFinite(c))) {
                throw new ArgumentOutOfRangeException(nameof(c), "Invalid scale parameter.");
            }

            Mu = mu;
            C = c;

            c_inv = 1d / c;
            bias = -2d / Pi * Log(c);
        }

        public double PDF(double x) {
            double u = (x - Mu) * c_inv + bias;

            if (IsNaN(u)) {
                return NaN;
            }
            if (IsInfinity(u)) {
                return 0d;
            }

            double pdf = PDFPade.Value(u) * c_inv;

            return pdf;
        }

        public double CDF(double x, Interval interval = Interval.Lower) {
            double u = (x - Mu) * c_inv + bias;

            if (IsNaN(u)) {
                return NaN;
            }

            double cdf = CDFPade.Value(u, interval != Interval.Lower);

            return cdf;
        }

        public double Quantile(double p, Interval interval = Interval.Lower) {
            if (!(p >= 0d && p <= 1d)) {
                return NaN;
            }

            double x = Mu + C * (QuantilePade.Value(p, interval != Interval.Lower) - bias);

            return x;
        }

        public double Sample(Random random) {
            double z = random.NextUniformOpenInterval01(), u = z - 0.5d;
            double w = random.NextUniformOpenInterval01();

            double r = 2d / Pi * (z * TanPi(u) * Pi - Log(Log(w) * CosPi(u) / (-2d * z * C)));
            double v = r * C + Mu;

            return v;
        }

        public bool Symmetric => false;

        public double Median => Mu + (median_base - bias) * C;

        public double Mode => Mu + (mode_base - bias) * C;

        public double Mean => NaN;

        public double Variance => NaN;

        public double Skewness => NaN;

        public double Kurtosis => NaN;

        public double Entropy => entropy_base + Log(C);

        public double Alpha => 1d;

        public double Beta => 1d;

        public static LandauDistribution operator +(LandauDistribution dist1, LandauDistribution dist2) {
            return new(dist1.Mu + dist2.Mu, dist1.C + dist2.C);
        }

        public static LandauDistribution operator -(LandauDistribution dist1, LandauDistribution dist2) {
            return new(dist1.Mu - dist2.Mu, dist1.C + dist2.C);
        }

        public static LandauDistribution operator +(LandauDistribution dist, double s) {
            return new(dist.Mu + s, dist.C);
        }

        public static LandauDistribution operator -(LandauDistribution dist, double s) {
            return new(dist.Mu - s, dist.C);
        }

        public static LandauDistribution operator *(LandauDistribution dist, double k) {
            return new((dist.Mu - 2d / Pi * dist.C * Log(k)) * k, dist.C * k);
        }

        public static LandauDistribution operator /(LandauDistribution dist, double k) {
            return new((dist.Mu + 2d / Pi * dist.C * Log(k)) / k, dist.C / k);
        }

        public override string ToString() {
            return $"{typeof(LandauDistribution).Name}[mu={Mu},c={C}]";
        }

        public string Formula => "p(x; mu, c) := stable_distribution(x; alpha = 1, beta = 1, mu, c)";

        private static class PDFPade {
            private static readonly double pi_half = ScaleB(Pi, -1);
            private static readonly double lambda_bias = 1.45158270528945486473;

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_0_1 = new(
                new ReadOnlyCollection<double>([
                    2.62240126375351657026e-1,
                    3.37943593381366824691e-1,
                    1.53537606095123787618e-1,
                    3.01423783265555668011e-2,
                    2.66982581491576132363e-3,
                    -1.57344124519315009970e-5,
                    3.46237168332264544791e-7,
                    2.54512306953704347532e-8,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.61596691542333069131e0,
                    1.31560197919990191004e0,
                    6.37865139714920275881e-1,
                    1.99051021258743986875e-1,
                    3.73788085017437528274e-2,
                    3.72580876403774116752e-3,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_1_2 = new(
                new ReadOnlyCollection<double>([
                    1.63531240868022603476e-1,
                    1.42818648212508067982e-1,
                    4.95816076364679661943e-2,
                    8.59234710489723831273e-3,
                    5.76649181954629544285e-4,
                    -5.66279925274108366994e-7,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.41478104966077351483e0,
                    9.41180365857002724714e-1,
                    3.65084346985789448244e-1,
                    8.77396986274371571301e-2,
                    1.24233749817860139205e-2,
                    8.57476298543168142524e-4,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_2_4 = new(
                new ReadOnlyCollection<double>([
                    9.55242261334771588094e-2,
                    6.66529732353979943139e-2,
                    1.80958840194356287100e-2,
                    2.34205449064047793618e-3,
                    1.16859089123286557482e-4,
                    -1.48761065213531458940e-7,
                    4.37245276130361710865e-9,
                    -8.10479404400603805292e-11,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.21670723402658089612e0,
                    6.58224466688607822769e-1,
                    2.00828142796698077403e-1,
                    3.64962053761472303153e-2,
                    3.76034152661165826061e-3,
                    1.74723754509505656326e-4,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_4_8 = new(
                new ReadOnlyCollection<double>([
                    3.83643820409470770350e-2,
                    1.97555000044256883088e-2,
                    3.71748668368617282698e-3,
                    3.04022677703754827113e-4,
                    8.76328889784070114569e-6,
                    -3.34900379044743745961e-9,
                    5.36581791174380716937e-11,
                    -5.50656207669255770963e-13,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    9.09290785092251223006e-1,
                    3.49404120360701349529e-1,
                    7.23730835206014275634e-2,
                    8.47875744543245845354e-3,
                    5.28021165718081084884e-4,
                    1.33941126695887244822e-5,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_8_16 = new(
                new ReadOnlyCollection<double>([
                    1.12656323880287532947e-2,
                    2.87311140580416132088e-3,
                    2.61788674390925516376e-4,
                    9.74096895307400300508e-6,
                    1.19317564431052244154e-7,
                    -6.99543778035110375565e-12,
                    4.33383971045699197233e-14,
                    -1.75185581239955717728e-16,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.94430267268436822392e-1,
                    1.00370783567964448346e-1,
                    1.05989564733662652696e-2,
                    6.04942184472254239897e-4,
                    1.72741008294864428917e-5,
                    1.85398104367945191152e-7,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_16_32 = new(
                new ReadOnlyCollection<double>([
                    2.83847488747490686627e-3,
                    4.95641151588714788287e-4,
                    2.79159792287747766415e-5,
                    5.93951761884139733619e-7,
                    3.89602689555407749477e-9,
                    -4.86595415551823027835e-14,
                    9.68524606019510324447e-17,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    3.01847536766892219351e-1,
                    3.63152433272831196527e-2,
                    2.20938897517130866817e-3,
                    7.05424834024833384294e-5,
                    1.09010608366510938768e-6,
                    6.08711307451776092405e-9,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_32_64 = new(
                new ReadOnlyCollection<double>([
                    6.85767880395157523315e-4,
                    4.08288098461672797376e-5,
                    8.10640732723079320426e-7,
                    6.10891161505083972565e-9,
                    1.37951861368789813737e-11,
                    -1.25906441382637535543e-17,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.23722380864018634550e-1,
                    6.05800403141772433527e-3,
                    1.47809654123655473551e-4,
                    1.84909364620926802201e-6,
                    1.08158235309005492372e-8,
                    2.16335841791921214702e-11,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_expp6_8 = new(
                new ReadOnlyCollection<double>([
                    6.78613480244945294595e-1,
                    9.61675759893298556080e-1,
                    3.45159462006746978086e-1,
                    6.32803373041761027814e-2,
                    6.93646175256407852991e-3,
                    4.69867700169714338273e-4,
                    1.76219117171149694118e-5,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.44693640094228656726e0,
                    5.46298626321591162873e-1,
                    1.01572892952421447864e-1,
                    1.04982575345680980744e-2,
                    7.65591730392359463367e-4,
                    2.69383817793665674679e-5,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_expp8_16 = new(
                new ReadOnlyCollection<double>([
                    6.51438485661317103070e-1,
                    2.67941671074735988081e-1,
                    5.18564629295719783781e-2,
                    6.18976337233135940231e-3,
                    5.08042228681335953236e-4,
                    2.97268230746003939324e-5,
                    1.24283200336057908183e-6,
                    3.35670921544537716055e-8,
                    5.06987792821954864905e-10,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.23792506680780833665e-1,
                    8.17040643791396371682e-2,
                    9.63961713981621216197e-3,
                    8.06584713485725204135e-4,
                    4.62050471704120102023e-5,
                    1.96919734048024406173e-6,
                    5.23890369587103685278e-8,
                    7.99399970089366802728e-10,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_expp16_32 = new(
                new ReadOnlyCollection<double>([
                    6.36745544906925230102e-1,
                    2.06319686601209029700e-1,
                    3.27498059700133287053e-2,
                    3.30913729536910108000e-3,
                    2.34809665750270531592e-4,
                    1.21234086846551635407e-5,
                    4.55253563898240922019e-7,
                    1.17544434819877511707e-8,
                    1.76754192209232807941e-10,
                    -2.78616504641875874275e-17,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    3.24145654925686670201e-1,
                    5.14350019501887110402e-2,
                    5.19867984016649969928e-3,
                    3.68798608372265018587e-4,
                    1.90449594112666257344e-5,
                    7.15068261954120746192e-7,
                    1.84646096630493837656e-8,
                    2.77636277083994601941e-10,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_expp32_64 = new(
                new ReadOnlyCollection<double>([
                    6.36619776379492082324e-1,
                    2.68158440168597706495e-1,
                    5.49040993767853738389e-2,
                    7.23458585096723552751e-3,
                    6.85438876301780090281e-4,
                    4.84561891424380633578e-5,
                    2.82092117716081590941e-6,
                    9.57557353473514565245e-8,
                    5.16773829224576217348e-9,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.21222294324039934056e-1,
                    8.62431574655015481812e-2,
                    1.13640608906815986975e-2,
                    1.07668486873466248474e-3,
                    7.61148039258802068270e-5,
                    4.43109262308946031382e-6,
                    1.50412757354817481381e-7,
                    8.11746432728995551732e-9,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_minus_1_0 = new(
                new ReadOnlyCollection<double>([
                    2.21762208692280384264e-1,
                    7.10041055270973473923e-1,
                    8.66556480457430718380e-1,
                    4.78718713740071686348e-1,
                    1.03670563650247405820e-1,
                    4.31699263023057628473e-3,
                    1.72029926636215817416e-3,
                    -2.76271972015177236271e-4,
                    1.89483904652983701680e-5,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    2.18155995697310361937e0,
                    2.53173077603836285217e0,
                    1.91802065831309251416e0,
                    9.94481663032480077373e-1,
                    3.72037148486473195054e-1,
                    8.85828240211801048938e-2,
                    1.41354784778520560313e-2,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_minus_2_1 = new(
                new ReadOnlyCollection<double>([
                    6.50763682207511020789e-3,
                    5.73790055136022120436e-2,
                    2.22375662069496257066e-1,
                    4.92288611166073916396e-1,
                    6.74552077334695078716e-1,
                    5.75057550963763663751e-1,
                    2.85690710485234671432e-1,
                    6.73776735655426117231e-2,
                    3.80321995712675339999e-3,
                    1.09503400950148681072e-3,
                    -9.00045301380982997382e-5,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.07919389927659014373e0,
                    2.56142472873207168042e0,
                    1.68357271228504881003e0,
                    2.23924151033591770613e0,
                    9.05629695159584880257e-1,
                    8.94372028246671579022e-1,
                    1.98616842716090037437e-1,
                    1.70142519339469434183e-1,
                    1.46288923980509020713e-2,
                    1.26171654901120724762e-2,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_minus_2_4 = new(
                new ReadOnlyCollection<double>([
                    6.31126317567898819465e-1,
                    5.28493759149515726917e-1,
                    3.28301410420682938866e-1,
                    1.31682639578153092699e-1,
                    3.86573798047656547423e-2,
                    7.77797337463414935830e-3,
                    9.97883658430364658707e-4,
                    6.05131104440018116255e-5,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    8.47781139548258655981e-1,
                    5.21797290075642096762e-1,
                    2.10939174293308469446e-1,
                    6.14856955543769263502e-2,
                    1.24427885618560158811e-2,
                    1.58973907730896566627e-3,
                    9.66647686344466292608e-5,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_minus_4_8 = new(
                new ReadOnlyCollection<double>([
                    6.26864481454444278646e-1,
                    5.10647753508714204745e-1,
                    1.98551443303285119497e-1,
                    4.71644854289800143386e-2,
                    7.71285919105951697285e-3,
                    8.93551020612017939395e-4,
                    6.97020145401946303751e-5,
                    4.17249760274638104772e-6,
                    7.73502439313710606153e-12,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    8.15124079722976906223e-1,
                    3.16755852188961901369e-1,
                    7.52819418000330690962e-2,
                    1.23053506566779662890e-2,
                    1.42615273721494498141e-3,
                    1.11211928184477279204e-4,
                    6.65899898061789485757e-6,
                ])
            );

            private static double PlusValue(double x) {
                Debug.Assert(x >= 0);

                if (x <= 64d) {
                    double y;

                    if (x <= 1d) {
                        Debug.WriteLine("pade minimum segment passed");

                        y = ApproxUtil.Pade(x, pade_plus_0_1);
                    }
                    else if (x <= 2d) {
                        y = ApproxUtil.Pade(x - 1d, pade_plus_1_2);
                    }
                    else if (x <= 4d) {
                        y = ApproxUtil.Pade(x - 2d, pade_plus_2_4);
                    }
                    else if (x <= 8d) {
                        y = ApproxUtil.Pade(x - 4d, pade_plus_4_8);
                    }
                    else if (x <= 16d) {
                        y = ApproxUtil.Pade(x - 8d, pade_plus_8_16);
                    }
                    else if (x <= 32d) {
                        y = ApproxUtil.Pade(x - 16d, pade_plus_16_32);
                    }
                    else {
                        y = ApproxUtil.Pade(x - 32d, pade_plus_32_64);
                    }

                    return y;
                }
                else {
                    int exponent = ILogB(x);

                    double v;
                    if (exponent < 8) {
                        v = ApproxUtil.Pade(Log2(ScaleB(x, -6)), pade_plus_expp6_8);
                    }
                    else if (exponent < 16) {
                        v = ApproxUtil.Pade(Log2(ScaleB(x, -8)), pade_plus_expp8_16);
                    }
                    else if (exponent < 32) {
                        v = ApproxUtil.Pade(Log2(ScaleB(x, -16)), pade_plus_expp16_32);
                    }
                    else if (exponent < 64) {
                        v = ApproxUtil.Pade(Log2(ScaleB(x, -32)), pade_plus_expp32_64);
                    }
                    else {
                        v = 2 / Pi;
                    }

                    double y = v / (x * x);

                    return y;
                }
            }

            private static double MinusValue(double x) {
                Debug.Assert(x <= 0);

                x = -x;

                if (x <= 2d) {
                    double y;
                    if (x <= 1d) {
                        y = ApproxUtil.Pade(1d - x, pade_minus_1_0);
                    }
                    else {
                        y = ApproxUtil.Pade(2d - x, pade_minus_2_1);
                    }

                    return y;
                }
                else if (x <= 8d) {
                    double v;
                    if (x <= 4d) {
                        v = ApproxUtil.Pade(x - 2d, pade_minus_2_4);
                    }
                    else {
                        v = ApproxUtil.Pade(x - 4d, pade_minus_4_8);
                    }

                    double sigma = Exp(x * pi_half - lambda_bias);

                    double y = v * Sqrt(sigma) * Exp(-sigma);

                    return y;
                }
                else {
                    return 0d;
                }
            }

            public static double Value(double x) {
                return (x >= 0d) ? PlusValue(x) : (x <= 0d) ? MinusValue(x) : NaN;
            }
        }

        private static class CDFPade {
            private static readonly double pi_half = ScaleB(Pi, -1);
            private static readonly double lambda_bias = 1.45158270528945486473;

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_0_1 = new(
                new ReadOnlyCollection<double>([
                    6.34761298487625202628e-1,
                    7.86558857265845597915e-1,
                    4.30220871807399303399e-1,
                    1.26410946316538340541e-1,
                    2.09346669713191648490e-2,
                    1.48926177023501002834e-3,
                    -5.93750588554108593271e-7,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.65227304522196452589e0,
                    1.29276828719607419526e0,
                    5.93815051307098615300e-1,
                    1.69165968013666952456e-1,
                    2.84272940328510367574e-2,
                    2.28001970477820696422e-3,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_1_2 = new(
                new ReadOnlyCollection<double>([
                    4.22133240358047652363e-1,
                    3.48421126689016131480e-1,
                    1.15402429637790321091e-1,
                    1.90374044978864005061e-2,
                    1.26628667888851698698e-3,
                    -5.75103242931559285281e-7,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.21277435324167238159e0,
                    6.38324046905267845243e-1,
                    1.81723381692749892660e-1,
                    2.80457012073363245106e-2,
                    1.93749385908189487538e-3,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_2_4 = new(
                new ReadOnlyCollection<double>([
                    2.95892137955791216378e-1,
                    2.29083899043580095868e-1,
                    7.09374171394372356009e-2,
                    1.08774274442674552229e-2,
                    7.69674715320139398655e-4,
                    1.63486840000680408991e-5,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.09704883482087441931e0,
                    5.10139057077147935327e-1,
                    1.27055234007499238241e-1,
                    1.74542139987310825683e-2,
                    1.18944143641885993718e-3,
                    2.55296292914537992309e-5,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_4_8 = new(
                new ReadOnlyCollection<double>([
                    1.73159318667565938776e-1,
                    6.95847424776057206679e-2,
                    1.04513924567165899506e-2,
                    6.35094718543965631442e-4,
                    1.04166111154771164657e-5,
                    1.43633490646363733467e-9,
                    -4.55493341295654514558e-11,
                    6.71119091495929467041e-13,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    6.23409270429130114247e-1,
                    1.54791925441839372663e-1,
                    1.85626981728559445893e-2,
                    1.01414235673220405086e-3,
                    1.63385654535791481980e-5,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_8_16 = new(
                new ReadOnlyCollection<double>([
                    8.90469147411748292410e-2,
                    2.76033447621178662228e-2,
                    3.26577485081539607943e-3,
                    1.77755752909150255339e-4,
                    4.20716551767396206445e-6,
                    3.19415703637929092564e-8,
                    -1.79900915228302845362e-13,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.36499987260915480890e-1,
                    7.67544181756713372678e-2,
                    6.83535263652329633233e-3,
                    3.15983778969051850073e-4,
                    6.84144567273078698399e-6,
                    5.00300197147417963939e-8,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_16_32 = new(
                new ReadOnlyCollection<double>([
                    4.35157264931262089762e-2,
                    8.46833474333913742597e-3,
                    6.43769318301002170686e-4,
                    2.39440197089740502223e-5,
                    4.45572968892675484685e-7,
                    3.76071815793351687179e-9,
                    1.04851094362145160445e-11,
                    -8.50646541795105885254e-18,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    2.59832721225510968607e-1,
                    2.75929030381330309762e-2,
                    1.53115657043391090526e-3,
                    4.70173086825204710446e-5,
                    7.76185172490852556883e-7,
                    6.10512879655564540102e-9,
                    1.64522607881748812093e-11,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_32_64 = new(
                new ReadOnlyCollection<double>([
                    2.11253031965493064317e-2,
                    1.36656844320536022509e-3,
                    2.99036224749763963099e-5,
                    2.54538665523638998222e-7,
                    6.79286608893558228264e-10,
                    -6.92803349600061706079e-16,
                    5.47233092767314029032e-19,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    9.71506209641408410168e-2,
                    3.52744690483830496158e-3,
                    5.85142319429623560735e-5,
                    4.29686638196055795330e-7,
                    1.06586221304077993137e-9,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_expp6_8 = new(
                new ReadOnlyCollection<double>([
                    6.60754766433212615409e-1,
                    2.47190065739055522599e-1,
                    4.17560046901040308267e-2,
                    3.71520821873148657971e-3,
                    2.03659383008528656781e-4,
                    2.52070598577347523483e-6,
                    -1.63741595848354479992e-8,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    3.92836792184266080580e-1,
                    6.64332913820571574875e-2,
                    5.59456053716889879620e-3,
                    3.44201583106671507027e-4,
                    2.74554105716911980435e-6,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_expp8_16 = new(
                new ReadOnlyCollection<double>([
                    6.44802371584831601817e-1,
                    2.74177359656349204309e-1,
                    5.53659240731871433983e-2,
                    6.97653365560511851744e-3,
                    6.17058143529799037402e-4,
                    3.94979574476108021136e-5,
                    1.88315864113369221822e-6,
                    6.10941845734962836501e-8,
                    1.39403332890347813312e-9,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.32345127287830884682e-1,
                    8.70500634789942065799e-2,
                    1.09253956356393590470e-2,
                    9.72576825490118007977e-4,
                    6.18656322285414147985e-5,
                    2.96375876501823390564e-6,
                    9.58622809886777038970e-8,
                    2.19059124630695181004e-9,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_expp16_32 = new(
                new ReadOnlyCollection<double>([
                    6.36685748306554972132e-1,
                    2.22217783148381285219e-1,
                    3.79173960692559280353e-2,
                    4.13394722917837684942e-3,
                    3.18141233442663766089e-4,
                    1.79745613243740552736e-5,
                    7.47632665728046334131e-7,
                    2.18258684729250152138e-8,
                    3.93038365129320422968e-10,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    3.49087806008685701060e-1,
                    5.95568283529034601477e-2,
                    6.49386742119035055908e-3,
                    4.99721374204563274865e-4,
                    2.82348248031305043777e-5,
                    1.17436903872210815656e-6,
                    3.42841159307801319359e-8,
                    6.17382517100568714012e-10,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_plus_expp32_64 = new(
                new ReadOnlyCollection<double>([
                    6.36619774420718062663e-1,
                    2.68594096777677177874e-1,
                    5.50713044649497737064e-2,
                    7.26574134143434960446e-3,
                    6.89173530168387629057e-4,
                    4.87688310559244353811e-5,
                    2.84218580121660744969e-6,
                    9.65240367429172366675e-8,
                    5.21722720068664704240e-9,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.21906621389193043384e-1,
                    8.65058026826346828750e-2,
                    1.14129998157398060009e-2,
                    1.08255124950652385121e-3,
                    7.66059006900869004871e-5,
                    4.46449501653114622960e-6,
                    1.51619602364037777665e-7,
                    8.19520132288940649002e-9,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_minus_1_0 = new(
                new ReadOnlyCollection<double>([
                    9.61609610406317335842e-2,
                    3.91836314722738553695e-1,
                    6.79862925205625107133e-1,
                    6.52516594941817706368e-1,
                    3.78594163612581127974e-1,
                    1.37741592243008345389e-1,
                    3.16100502353317199197e-2,
                    3.94935603975622336575e-3,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.76863983252615276767e0,
                    1.81486018095087241378e0,
                    1.17295504548962999723e0,
                    5.33998066342362562313e-1,
                    1.66508320794082632235e-1,
                    3.42192028846565504290e-2,
                    3.94691613177524994796e-3,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_minus_2_1 = new(
                new ReadOnlyCollection<double>([
                    7.07114056489178077423e-4,
                    7.35277969197058909845e-3,
                    3.45402694579204809691e-2,
                    9.62849773112695332289e-2,
                    1.75738736725818007992e-1,
                    2.18309266582058485951e-1,
                    1.85680388782727289455e-1,
                    1.06177394398691169291e-1,
                    3.94880388335722224211e-2,
                    9.46543177731050647162e-3,
                    1.50949646857411896396e-3,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.19520021153535414164e0,
                    2.24057032777744601624e0,
                    1.63635577968560162720e0,
                    1.58952087228427876880e0,
                    7.63062254749311648018e-1,
                    4.65805990343825931327e-1,
                    1.45821531714775598887e-1,
                    5.42393925507104531351e-2,
                    9.84276292481407168381e-3,
                    1.54787649925009672534e-3,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_minus_2_4 = new(
                new ReadOnlyCollection<double>([
                    3.71658823632747235572e-1,
                    2.81493346318174084721e-1,
                    1.80052521696460721846e-1,
                    7.65907659636944822120e-2,
                    2.33352148213280934280e-2,
                    5.02308701022480574067e-3,
                    6.29239919421134075502e-4,
                    8.36993181707604609065e-6,
                    -8.38295154747385945293e-6,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    6.62107509936390708604e-1,
                    4.72501892305147483696e-1,
                    1.84446743813050604353e-1,
                    5.99971792581573339487e-2,
                    1.24751029844082800143e-2,
                    1.56705297654475773870e-3,
                    2.36392472352050487445e-5,
                    -2.11667044716450080820e-5,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_minus_4_8 = new(
                new ReadOnlyCollection<double>([
                    3.97500903816385095134e-1,
                    5.08559630146730380854e-1,
                    2.99190443368166803486e-1,
                    1.07339363365158174786e-1,
                    2.61694301269384158162e-2,
                    4.58386867966451237870e-3,
                    5.80610284231484509069e-4,
                    5.07249042503156949021e-5,
                    2.91644292826084281875e-6,
                    9.75453868235609527534e-12,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.27376091725485414303e0,
                    7.49829208702328578188e-1,
                    2.69157374996960976399e-1,
                    6.55795320040378662663e-2,
                    1.14912646428788757804e-2,
                    1.45541420582309879973e-3,
                    1.27135040794481871472e-4,
                    7.31138551538712031061e-6,
                ])
            );

            private static double PlusValue(double x) {
                Debug.Assert(x >= 0);

                if (x <= 64d) {
                    double y;

                    if (x <= 1d) {
                        Debug.WriteLine("pade minimum segment passed");

                        y = ApproxUtil.Pade(x, pade_plus_0_1);
                    }
                    else if (x <= 2d) {
                        y = ApproxUtil.Pade(x - 1d, pade_plus_1_2);
                    }
                    else if (x <= 4d) {
                        y = ApproxUtil.Pade(x - 2d, pade_plus_2_4);
                    }
                    else if (x <= 8d) {
                        y = ApproxUtil.Pade(x - 4d, pade_plus_4_8);
                    }
                    else if (x <= 16d) {
                        y = ApproxUtil.Pade(x - 8d, pade_plus_8_16);
                    }
                    else if (x <= 32d) {
                        y = ApproxUtil.Pade(x - 16d, pade_plus_16_32);
                    }
                    else {
                        y = ApproxUtil.Pade(x - 32d, pade_plus_32_64);
                    }

                    return y;
                }
                else {
                    int exponent = ILogB(x);

                    double v;
                    if (exponent < 8) {
                        v = ApproxUtil.Pade(Log2(ScaleB(x, -6)), pade_plus_expp6_8);
                    }
                    else if (exponent < 16) {
                        v = ApproxUtil.Pade(Log2(ScaleB(x, -8)), pade_plus_expp8_16);
                    }
                    else if (exponent < 32) {
                        v = ApproxUtil.Pade(Log2(ScaleB(x, -16)), pade_plus_expp16_32);
                    }
                    else if (exponent < 64) {
                        v = ApproxUtil.Pade(Log2(ScaleB(x, -32)), pade_plus_expp32_64);
                    }
                    else {
                        v = 2 / Pi;
                    }

                    double y = v / x;

                    return y;
                }
            }

            private static double MinusValue(double x) {
                Debug.Assert(x <= 0);

                x = -x;

                if (x <= 2d) {
                    double y;
                    if (x <= 1d) {
                        y = ApproxUtil.Pade(1d - x, pade_minus_1_0);
                    }
                    else {
                        y = ApproxUtil.Pade(2d - x, pade_minus_2_1);
                    }

                    return y;
                }
                else if (x <= 8d) {
                    double v;
                    if (x <= 4d) {
                        v = ApproxUtil.Pade(x - 2d, pade_minus_2_4);
                    }
                    else {
                        v = ApproxUtil.Pade(x - 4d, pade_minus_4_8);
                    }

                    double sigma = Exp(x * pi_half - lambda_bias);

                    double y = v / Sqrt(sigma) * Exp(-sigma);

                    return y;
                }
                else {
                    return 0d;
                }
            }

            public static double Value(double x, bool complementary) {
                if (x >= 0d) {
                    return complementary ? PlusValue(x) : 1d - PlusValue(x);
                }
                else if (x <= 0d) {
                    return complementary ? 1d - MinusValue(x) : MinusValue(x);
                }
                else {
                    return NaN;
                }
            }
        }

        private static class QuantilePade {
            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_upper_0p125_0p25 = new(
                new ReadOnlyCollection<double>([
                    5.68160868054034111703e0,
                    1.06098927525586705381e2,
                    5.74509518025029027944e2,
                    4.91117375866809056969e2,
                    -2.92607000654635606895e3,
                    -3.82912009541683403499e3,
                    2.49195208452006100935e3,
                    1.29413301335116683836e3,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    2.69603865809599480308e1,
                    2.63378422475372461819e2,
                    1.09903493506098212946e3,
                    1.60315072092792425370e3,
                    -5.44710468198458322870e2,
                    -1.76410218726878681387e3,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_upper_0p25_0p375 = new(
                new ReadOnlyCollection<double>([
                    2.55081568282045924981e0,
                    5.38750533719526696218e0,
                    -2.32797421725187349036e1,
                    -3.96043566411306749784e1,
                    3.80609941977115436545e1,
                    3.35014421131920266346e1,
                    -1.17490458743273503838e1,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    7.52439409918350484765e0,
                    1.34784954182866689668e1,
                    -9.21002543625052363446e0,
                    -2.67378141317474265949e1,
                    2.10158795079902783094e0,
                    5.90098096212203282798e0,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_upper_0p375_0p5 = new(
                new ReadOnlyCollection<double>([
                    1.31348919222343858178e0,
                    -1.06646675961352786791e0,
                    -1.80946160022120488884e1,
                    -1.53457017598330440033e0,
                    4.71260102173048370028e1,
                    4.61048467818771410732e0,
                    -2.80957284947853532418e1,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.71007453129016317772e0,
                    1.31946404969596908872e0,
                    -1.70321827414586880227e1,
                    -1.11253495615474018666e1,
                    1.62659086449959446986e1,
                    7.37109203295032098763e0,
                    -2.43898047338699777337e0,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_upper_expm3_4 = new(
                new ReadOnlyCollection<double>([
                    7.10201085067542566037e-1,
                    6.70042401812679849451e-1,
                    2.42799404088685074098e-1,
                    4.80613880364042262227e-2,
                    6.04473313360581797461e-3,
                    5.09172911021654842046e-4,
                    -6.63145317984529265677e-6,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    9.18649629646213969612e-1,
                    3.66343989541898286306e-1,
                    8.01010534748206001446e-2,
                    1.00553335007168823115e-2,
                    6.30966763237332075752e-4,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_upper_expm4_8 = new(
                new ReadOnlyCollection<double>([
                    7.06147398566773538296e-1,
                    4.26802162741800814387e-1,
                    1.32254436707168800420e-1,
                    2.86055054496737936396e-2,
                    3.63373131686703931514e-3,
                    3.84438945816411937013e-4,
                    1.67768561420296743529e-5,
                    8.76982374043363061978e-7,
                    -1.99744396595921347207e-8,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    6.28190787856605587324e-1,
                    2.10992746593815791546e-1,
                    4.44397672327578790713e-2,
                    6.02768341661155914525e-3,
                    5.46578619531721658923e-4,
                    3.11116573895074296750e-5,
                    1.17729007979018602786e-6,
                    -2.78441865351376040812e-8,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_upper_expm8_16 = new(
                new ReadOnlyCollection<double>([
                    6.48209596014908359251e-1,
                    2.52611824671691390768e-1,
                    4.65114070477803399291e-2,
                    5.23373513313686849909e-3,
                    3.83113384161076881958e-4,
                    1.96230077517629530809e-5,
                    5.83117485120890819338e-7,
                    6.92614450423703079737e-9,
                    -3.89531123166658723619e-10,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    3.99413988076189200840e-1,
                    7.32068638518417765776e-2,
                    8.15517102642752348889e-3,
                    6.09126071418098074914e-4,
                    3.03794079468789962611e-5,
                    9.32109079205017197662e-7,
                    1.05435710482490499583e-8,
                    -6.08748435983193979360e-10,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_upper_expm16_32 = new(
                new ReadOnlyCollection<double>([
                    6.36719010559816164896e-1,
                    2.06504115804034148753e-1,
                    3.28085429275407182582e-2,
                    3.31676417519020335859e-3,
                    2.35502578757551086372e-4,
                    1.21652240566662139418e-5,
                    4.57039495420392748658e-7,
                    1.18090959236399583940e-8,
                    1.77492646969597480221e-10,
                    -2.19331267300885448673e-17,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    3.24422807416528490276e-1,
                    5.15290129833049138552e-2,
                    5.21051235888272287209e-3,
                    3.69895399249472399625e-4,
                    1.91103139437893226482e-5,
                    7.17882574725373091636e-7,
                    1.85502934977316481559e-8,
                    2.78798057565507249164e-10,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_upper_expm32_64 = new(
                new ReadOnlyCollection<double>([
                    6.36619775525705206992e-1,
                    2.68335698140634792041e-1,
                    5.49803347535070103650e-2,
                    7.25018344556356907109e-3,
                    6.87753481255849254220e-4,
                    4.86155006277788340253e-5,
                    2.84604768310787862450e-6,
                    9.56133960810049319917e-8,
                    5.26850116571886385248e-9,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.21500730173440590900e-1,
                    8.63629077498258325752e-2,
                    1.13885615328098640032e-2,
                    1.08032064178130906887e-3,
                    7.63650498196064792408e-5,
                    4.47056124637379045275e-6,
                    1.50189171357721423127e-7,
                    8.27574227882033707932e-9,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_0p125_0p25 = new(
                new ReadOnlyCollection<double>([
                    -8.77109518013577785811e-1,
                    -1.03442936529923615496e1,
                    -1.03389868296950570121e1,
                    2.01575691867458616553e2,
                    4.59115079925618829199e2,
                    -3.38676271744958577802e2,
                    -5.38213647878547918506e2,
                    1.99214574934960143349e2,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.64177607733998839003e1,
                    8.10042194014991761178e1,
                    7.61952772645589839171e1,
                    -2.52698871224510918595e2,
                    -1.95365983250723202416e2,
                    2.61928845964255538379e2,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_0p25_0p375 = new(
                new ReadOnlyCollection<double>([
                    -4.17764764050720190117e-1,
                    1.27887601021900963655e0,
                    1.80329928265996817279e1,
                    2.35783605878556791719e1,
                    -2.67160590411398800149e1,
                    -2.36192101013335692266e1,
                    8.30396110938939237358e0,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    5.37459525158081633669e0,
                    2.35696607501498012129e0,
                    -1.71117034150268575909e1,
                    -6.72278235529877170403e0,
                    1.27763043804603299034e1,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_0p375_0p5 = new(
                new ReadOnlyCollection<double>([
                    3.74557416577759554506e-2,
                    3.87808262376545756299e0,
                    4.03092288183382979104e0,
                    -1.65221829710249468257e1,
                    -6.99689838230114367276e0,
                    1.51123479911771488314e1,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.37863773851525662884e-1,
                    -6.35020262707816744534e0,
                    3.07646508389502660442e-1,
                    9.72566583784248877260e0,
                    -2.72338088170674280735e0,
                    -1.58608957980133006476e0,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_expm3_4 = new(
                new ReadOnlyCollection<double>([
                    -8.77109518013577852585e-1,
                    -1.08703720146608358678e0,
                    -4.34198537684719253325e-1,
                    -6.97264194535092564620e-2,
                    -4.20721933993302797971e-3,
                    -6.27420063107527426396e-5,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    8.38688797993971740640e-1,
                    2.47558526682310722526e-1,
                    3.03952783355954712472e-2,
                    1.39226078796010665644e-3,
                    1.43993679246435688244e-5,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_expm4_8 = new(
                new ReadOnlyCollection<double>([
                    -1.16727296241754548410e0,
                    -1.12325365855062172009e0,
                    -3.96403456954867129566e-1,
                    -6.50024588048629862189e-2,
                    -5.08582387678609504048e-3,
                    -1.71657051345258316598e-4,
                    -1.81536405273085024830e-6,
                    -9.65262938333207656548e-10,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    7.55271574611337871389e-1,
                    2.16323131117540100488e-1,
                    2.92693206540519768049e-2,
                    1.89396907936678571916e-3,
                    5.20017914327360594265e-5,
                    4.18896774212993675707e-7,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_expm8_16 = new(
                new ReadOnlyCollection<double>([
                    -1.78348038398799868409e0,
                    -7.74779087785346936524e-1,
                    -1.27121601027522656374e-1,
                    -9.86675785835385622362e-3,
                    -3.69510132425310943600e-4,
                    -6.00811940375633438805e-6,
                    -3.06397799506512676163e-8,
                    -7.34821360521886161256e-12,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    3.76606062137668223823e-1,
                    5.37821995022686641494e-2,
                    3.62736078766811383733e-3,
                    1.16954398984720362997e-4,
                    1.59917906784160311385e-6,
                    6.41144889614705503307e-9,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_expm16_32 = new(
                new ReadOnlyCollection<double>([
                    -2.32474749499506229415e0,
                    -4.81681429397597263092e-1,
                    -3.79696253130015182335e-2,
                    -1.42328672650093755545e-3,
                    -2.58335052925986849305e-5,
                    -2.03945574260603170161e-7,
                    -5.04229972664978604816e-10,
                    -5.49506755992282162712e-14,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.87186049570056737301e-1,
                    1.32852903862611979806e-2,
                    4.45262195863310928309e-4,
                    7.13306978839226580931e-6,
                    4.84555343060572391776e-8,
                    9.65086092007764297450e-11,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_expm32_64 = new(
                new ReadOnlyCollection<double>([
                    -2.82318656228158372998e0,
                    -2.84346379198027589453e-1,
                    -1.09194719815749710073e-2,
                    -1.99728160102967185378e-4,
                    -1.77069359938827653381e-6,
                    -6.82828539186572955883e-9,
                    -8.22634582905944543176e-12,
                    -4.10585514777842307175e-16,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    9.29910333991046040738e-2,
                    3.27860300729204691815e-3,
                    5.45852206475929614010e-5,
                    4.34395271645812189497e-7,
                    1.46600782366946777467e-9,
                    1.45083131237841500574e-12,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_expm64_128 = new(
                new ReadOnlyCollection<double>([
                    -3.29700011190686231229e0,
                    -1.62920309130909343601e-1,
                    -3.07152472866757852259e-3,
                    -2.75922040607620211449e-5,
                    -1.20144242264703283024e-7,
                    -2.27410079849018964454e-10,
                    -1.34109445298156050256e-13,
                    -3.08843378675512185582e-18,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    4.62324092774919223927e-2,
                    8.10410923007867515072e-4,
                    6.70843016241177926470e-6,
                    2.65459014339231700938e-8,
                    4.45531791525831169724e-11,
                    2.19324401673412172456e-14,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_expm128_256 = new(
                new ReadOnlyCollection<double>([
                    -3.75666995985336008568e0,
                    -9.15751436135409108392e-2,
                    -8.51745858385908954959e-4,
                    -3.77453552696508401182e-6,
                    -8.10504146884381804474e-9,
                    -7.55871397276946580837e-12,
                    -2.19023097542770265117e-15,
                    -2.34270094396556916060e-20,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    2.30119177073875808729e-2,
                    2.00787377759037971795e-4,
                    8.27382543511838001513e-7,
                    1.62997898759733931959e-9,
                    1.36215810410261098317e-12,
                    3.33957268115953023683e-16,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_expm256_512 = new(
                new ReadOnlyCollection<double>([
                    -4.20826069989721597050e0,
                    -5.07864788729928381957e-2,
                    -2.33825872475869133650e-4,
                    -5.12795917403072758309e-7,
                    -5.44657955194364350768e-10,
                    -2.51001805474510910538e-13,
                    -3.58448226638949307172e-17,
                    -1.79092368272097571876e-22,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    1.14671758705641048135e-2,
                    4.98614103841229871806e-5,
                    1.02397186002860292625e-7,
                    1.00544286633906421384e-10,
                    4.18843275058038084849e-14,
                    5.11960642868907665857e-18,
                ])
            );

            private static readonly (ReadOnlyCollection<double> numer, ReadOnlyCollection<double> denom) pade_lower_expm512_1024 = new(
                new ReadOnlyCollection<double>([
                    -4.65527239540648658214e0,
                    -2.78834161568280967534e-2,
                    -6.37014695368461940922e-5,
                    -6.92971221299243529202e-8,
                    -3.64900562915285147191e-11,
                    -8.32868843440595945586e-15,
                    -5.87602374631705229119e-19,
                    -1.37812578498484605190e-24,
                ]),
                new ReadOnlyCollection<double>([
                    1.00000000000000000000e0,
                    5.72000087046224585566e-3,
                    1.24068329655043560901e-5,
                    1.27105410419102416943e-8,
                    6.22649556008196699310e-12,
                    1.29416254332222127404e-15,
                    7.89365027125866583275e-20,
                ])
            );

            private static double UpperValue(double x) {
                Debug.Assert(x <= 0.5d);

                if (x >= 0.125d) {
                    double y;
                    if (x <= 0.25d) {
                        y = ApproxUtil.Pade(x - 0.125d, pade_upper_0p125_0p25);
                    }
                    else if (x <= 0.375d) {
                        y = ApproxUtil.Pade(x - 0.25d, pade_upper_0p25_0p375);
                    }
                    else {
                        y = ApproxUtil.Pade(x - 0.375d, pade_upper_0p375_0p5);
                    }

                    return y;
                }
                else {
                    double v;
                    int exponent = ILogB(x);

                    if (exponent >= -4) {
                        v = ApproxUtil.Pade(-Log2(ScaleB(x, 3)), pade_upper_expm3_4);
                    }
                    else if (exponent >= -8) {
                        v = ApproxUtil.Pade(-Log2(ScaleB(x, 4)), pade_upper_expm4_8);
                    }
                    else if (exponent >= -16) {
                        v = ApproxUtil.Pade(-Log2(ScaleB(x, 8)), pade_upper_expm8_16);
                    }
                    else if (exponent >= -32) {
                        v = ApproxUtil.Pade(-Log2(ScaleB(x, 16)), pade_upper_expm16_32);
                    }
                    else if (exponent >= -64) {
                        v = ApproxUtil.Pade(-Log2(ScaleB(x, 32)), pade_upper_expm32_64);
                    }
                    else {
                        v = ScaleB(1 / Pi, 1);
                    }

                    double y = v / x;

                    return y;
                }
            }

            private static double LowerValue(double x) {
                Debug.Assert(x <= 0.5d);

                if (x >= 0.125d) {
                    double y;
                    if (x <= 0.25d) {
                        y = ApproxUtil.Pade(x - 0.125d, pade_lower_0p125_0p25);
                    }
                    else if (x <= 0.375d) {
                        y = ApproxUtil.Pade(x - 0.25d, pade_lower_0p25_0p375);
                    }
                    else {
                        y = ApproxUtil.Pade(x - 0.375d, pade_lower_0p375_0p5);
                    }

                    return y;
                }
                else {
                    double y;
                    int exponent = ILogB(x);

                    if (exponent >= -4) {
                        y = ApproxUtil.Pade(-Log2(ScaleB(x, 3)), pade_lower_expm3_4);
                    }
                    else if (exponent >= -8) {
                        y = ApproxUtil.Pade(-Log2(ScaleB(x, 4)), pade_lower_expm4_8);
                    }
                    else if (exponent >= -16) {
                        y = ApproxUtil.Pade(-Log2(ScaleB(x, 8)), pade_lower_expm8_16);
                    }
                    else if (exponent >= -32) {
                        y = ApproxUtil.Pade(-Log2(ScaleB(x, 16)), pade_lower_expm16_32);
                    }
                    else if (exponent >= -64) {
                        y = ApproxUtil.Pade(-Log2(ScaleB(x, 32)), pade_lower_expm32_64);
                    }
                    else if (exponent >= -128) {
                        y = ApproxUtil.Pade(-Log2(ScaleB(x, 64)), pade_lower_expm64_128);
                    }
                    else if (exponent >= -256) {
                        y = ApproxUtil.Pade(-Log2(ScaleB(x, 128)), pade_lower_expm128_256);
                    }
                    else if (exponent >= -512) {
                        y = ApproxUtil.Pade(-Log2(ScaleB(x, 256)), pade_lower_expm256_512);
                    }
                    else if (exponent >= -1024) {
                        y = ApproxUtil.Pade(-Log2(ScaleB(x, 512)), pade_lower_expm512_1024);
                    }
                    else {
                        return NegativeInfinity;
                    }

                    return y;
                }
            }

            public static double Value(double x, bool complementary) {
                if (x > 0.5) {
                    return Value(1d - x, !complementary);
                }

                return complementary ? UpperValue(x) : LowerValue(x);
            }
        }
    }
}
