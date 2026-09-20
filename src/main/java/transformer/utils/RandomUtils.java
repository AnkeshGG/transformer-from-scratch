/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.utils;

import java.util.Random;

/**
 * RandomUtils — weight initialisation and seeded random number generation
 * for the Transformer implementation.
 *
 * <h2>Xavier / Glorot Initialisation</h2>
 * <p>For a linear layer with {@code fanIn} inputs and {@code fanOut} outputs,
 * Xavier initialisation draws weights from:
 * <pre>
 *   W ~ Uniform(−limit, +limit)
 *   where limit = sqrt(6 / (fanIn + fanOut))
 * </pre>
 *
 * <p>This choice preserves the variance of activations through the forward pass
 * and the variance of gradients through the backward pass, helping avoid
 * vanishing / exploding gradients at initialisation.
 *
 * <h2>Alternative: Normal Xavier</h2>
 * <pre>
 *   W ~ Normal(0, sqrt(2 / (fanIn + fanOut)))
 * </pre>
 *
 * <p>We use the <em>uniform</em> variant for embedding and projection matrices.
 */
public final class RandomUtils {

    private RandomUtils() {} // Utility class — no instantiation

    /**
     * Fill a 2-D float array with Xavier-uniform initialised values.
     *
     * <p>Xavier uniform: W[i][j] ~ Uniform(−limit, +limit)
     * where limit = sqrt(6 / (fanIn + fanOut)).
     *
     * @param weights 2-D array to fill (rows = fanOut, cols = fanIn)
     * @param fanIn   number of input units (typically weights[0].length)
     * @param fanOut  number of output units (typically weights.length)
     * @param rng     seeded random number generator for reproducibility
     */
    public static void xavierUniform(float[][] weights, int fanIn, int fanOut, Random rng) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        for (float[] row : weights)
            for (int c = 0; c < row.length; c++)
                row[c] = (float) (rng.nextDouble() * 2.0 * limit - limit);
    }

    /**
     * Fill a 1-D float array with Xavier-uniform initialised values.
     * Uses the same formula with fanIn and fanOut being the total size.
     */
    public static void xavierUniform(float[] weights, int fanIn, int fanOut, Random rng) {
        double limit = Math.sqrt(6.0 / (fanIn + fanOut));
        for (int i = 0; i < weights.length; i++)
            weights[i] = (float) (rng.nextDouble() * 2.0 * limit - limit);
    }

    /**
     * Fill a 2-D float array with small normal random values (mean=0, std=stddev).
     * Useful for embedding tables.
     *
     * @param weights 2-D array to fill
     * @param stddev  standard deviation
     * @param rng     seeded random number generator
     */
    public static void normalInit(float[][] weights, double stddev, Random rng) {
        for (float[] row : weights)
            for (int c = 0; c < row.length; c++)
                row[c] = (float) (rng.nextGaussian() * stddev);
    }

    /**
     * Fill a 1-D float array with zeros.
     * Used for bias initialisation (biases are typically zero-initialised).
     */
    public static void zeroInit(float[] array) {
        java.util.Arrays.fill(array, 0.0f);
    }

    /**
     * Fill a 1-D float array with ones.
     * Used for LayerNorm gamma (scale) initialisation.
     */
    public static void oneInit(float[] array) {
        java.util.Arrays.fill(array, 1.0f);
    }

    /**
     * Create a seeded Random from a long seed.
     * Pass the seed from {@link transformer.TransformerConfig} for reproducibility.
     */
    public static Random createRng(long seed) {
        return new Random(seed);
    }
}
