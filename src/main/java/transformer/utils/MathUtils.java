/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.utils;

/**
 * MathUtils — stateless utility methods for common mathematical operations
 * used across the Transformer implementation.
 *
 * <p>All methods are static to enable convenient use without instantiation.
 */
public final class MathUtils {

    private MathUtils() {} // Utility class — no instantiation

    // ── Stable Softmax ───────────────────────────────────────────────────────

    /**
     * Compute numerically stable softmax over a 1-D float array.
     *
     * <p>Standard softmax:
     * <pre>
     *   softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
     * </pre>
     *
     * <p>Numerically stable version (subtract max first):
     * <pre>
     *   m = max(x)
     *   softmax(x)_i = exp(x_i − m) / Σ_j exp(x_j − m)
     * </pre>
     *
     * <p>The max subtraction does NOT change the result mathematically:
     * <pre>
     *   exp(x_i − m) / Σ exp(x_j − m)
     *   = [exp(x_i) / exp(m)] / [Σ exp(x_j) / exp(m)]
     *   = exp(x_i) / Σ exp(x_j)
     * </pre>
     * but prevents {@code exp(x_i)} from overflowing to +∞.
     *
     * @param x input array (not modified)
     * @return new array with softmax applied
     */
    public static float[] softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) if (v > max) max = v;

        double sum = 0.0;
        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = (float) Math.exp(x[i] - max);
            sum += result[i];
        }
        for (int i = 0; i < result.length; i++)
            result[i] /= (float) sum;
        return result;
    }

    /**
     * Compute numerically stable log-softmax over a 1-D float array.
     *
     * <p>log-softmax(x)_i = x_i − log(Σ_j exp(x_j))
     * Stable version:
     * <pre>
     *   m = max(x)
     *   log-softmax(x)_i = (x_i − m) − log(Σ_j exp(x_j − m))
     * </pre>
     *
     * <p>Used in cross-entropy loss to avoid computing softmax → log separately.
     *
     * @param x input array (not modified)
     * @return new array with log-softmax applied
     */
    public static float[] logSoftmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) if (v > max) max = v;

        double sumExp = 0.0;
        for (float v : x) sumExp += Math.exp(v - max);
        double logSumExp = Math.log(sumExp);

        float[] result = new float[x.length];
        for (int i = 0; i < x.length; i++)
            result[i] = (float) ((x[i] - max) - logSumExp);
        return result;
    }

    // ── Clipping ─────────────────────────────────────────────────────────────

    /**
     * Clip a value to the range [min, max].
     */
    public static float clip(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    // ── Array argmax ─────────────────────────────────────────────────────────

    /**
     * Return the index of the maximum value in the array.
     * Used in greedy decoding: pick the token with the highest probability.
     */
    public static int argmax(float[] array) {
        int bestIdx = 0;
        float bestVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > bestVal) {
                bestVal = array[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // ── Array printing ────────────────────────────────────────────────────────

    /**
     * Format a float array for display (useful in debug output).
     */
    public static String formatArray(float[] arr, int maxElements) {
        StringBuilder sb = new StringBuilder("[");
        int limit = Math.min(arr.length, maxElements);
        for (int i = 0; i < limit; i++) {
            sb.append(String.format("%.4f", arr[i]));
            if (i < limit - 1) sb.append(", ");
        }
        if (arr.length > maxElements) sb.append(", ...");
        sb.append("]");
        return sb.toString();
    }
}
