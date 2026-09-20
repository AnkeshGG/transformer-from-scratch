/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.activation;

/**
 * ActivationFunctions — static activation functions used in the Transformer.
 *
 * <p>The feed-forward sublayer uses an activation function between its two linear
 * projections. The original paper uses ReLU; modern variants (GPT, BERT) use GELU.
 * This class provides both, plus a clean enum for selecting between them.
 */
public final class ActivationFunctions {

    private ActivationFunctions() {}

    /** Supported activation function types. */
    public enum Type {
        /** Rectified Linear Unit: max(0, x). Simple and fast. */
        RELU,
        /**
         * Gaussian Error Linear Unit.
         * GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
         * Smoother than ReLU; used in BERT, GPT.
         */
        GELU
    }

    /**
     * Apply element-wise ReLU to a float array.
     * ReLU(x) = max(0, x)
     *
     * @param x input array (not modified)
     * @return new array with ReLU applied
     */
    public static float[] relu(float[] x) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++)
            out[i] = Math.max(0f, x[i]);
        return out;
    }

    /**
     * Apply element-wise GELU to a float array.
     * Uses the tanh approximation from the original GELU paper.
     *
     * GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
     *
     * @param x input array (not modified)
     * @return new array with GELU applied
     */
    public static float[] gelu(float[] x) {
        final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            double xi = x[i];
            out[i] = (float) (0.5 * xi *
                (1.0 + Math.tanh(SQRT_2_OVER_PI * (xi + 0.044715 * xi * xi * xi))));
        }
        return out;
    }

    /**
     * Apply a named activation function to a float array.
     *
     * @param x    input array
     * @param type activation type
     * @return activated array
     */
    public static float[] apply(float[] x, Type type) {
        return switch (type) {
            case RELU -> relu(x);
            case GELU -> gelu(x);
        };
    }

    /**
     * Apply element-wise ReLU to a 2-D float array (Matrix data representation).
     * Used directly by FeedForward for efficiency.
     */
    public static float[][] relu2D(float[][] x) {
        float[][] out = new float[x.length][x[0].length];
        for (int r = 0; r < x.length; r++)
            for (int c = 0; c < x[r].length; c++)
                out[r][c] = Math.max(0f, x[r][c]);
        return out;
    }
}
