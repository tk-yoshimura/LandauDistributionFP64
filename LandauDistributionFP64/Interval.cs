namespace LandauDistributionFP64 {
    public enum Interval {
        Lower,
        Upper,
        Complementary = Upper,
        NegativeInfinityToX = Lower,
        XToPositiveInfinity = Upper,
    }
}
