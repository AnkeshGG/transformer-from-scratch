/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.attention;

/**
 * AttentionMask — Factory class for creating attention masks used in Transformer models.
 *
 * <h2>Two types of masks</h2>
 *
 * <h3>1. Causal (decoder self-attention) mask</h3>
 * <p>In autoregressive generation, position {@code i} must NOT be able to attend
 * to any future position {@code j > i}. This is enforced by the causal mask.
 *
 * <p>The mask is added to the attention scores (before softmax) as:
 * <pre>
 *   mask[i][j] = 0      if j ≤ i  (allowed: attend to past and present)
 *   mask[i][j] = −∞     if j > i  (blocked: do not attend to future)
 * </pre>
 *
 * <p>When this is added to the scores and then softmax is applied:
 * <ul>
 *   <li>Positions with score → −∞ produce exp(−∞) = 0 after softmax.
 *   <li>So those positions contribute nothing to the weighted sum.
 * </ul>
 *
 * <p>Example for sequence length 4:
 * <pre>
 *   Position:   j=0   j=1   j=2   j=3
 *   i=0:      [  0,   −∞,   −∞,   −∞ ]  (only attends to itself)
 *   i=1:      [  0,    0,   −∞,   −∞ ]  (attends to positions 0 and 1)
 *   i=2:      [  0,    0,    0,   −∞ ]
 *   i=3:      [  0,    0,    0,    0 ]  (attends to all past positions)
 * </pre>
 *
 * <h3>2. Padding mask</h3>
 * <p>Padding tokens (ID = 0 by convention) should not be attended to.
 * A padding mask blocks those positions with −∞.
 *
 * <h2>Implementation note</h2>
 * <p>Masks are returned as {@code float[][]} and directly added to the
 * attention scores matrix before softmax. Using {@code Float.NEGATIVE_INFINITY}
 * for blocked positions ensures exp(−∞) = 0 exactly (or very close to zero
 * due to floating-point).
 */
public final class AttentionMask {

    /** Value used to block attention (added to score before softmax → effectively 0 probability). */
    public static final float MASK_VALUE = Float.NEGATIVE_INFINITY;

    private AttentionMask() {} // Static factory — no instantiation

    // ── Causal mask ──────────────────────────────────────────────────────────

    /**
     * Create a causal (upper-triangular) mask for decoder self-attention.
     *
     * <p>Returns a [seqLen × seqLen] float matrix where:
     * <pre>
     *   mask[i][j] = 0      if j ≤ i  (past/present → allowed)
     *   mask[i][j] = -∞     if j > i  (future → blocked)
     * </pre>
     *
     * <p>This mask is added to the raw attention scores before softmax,
     * so softmax outputs ≈ 0 for all future positions.
     *
     * @param seqLen sequence length (T)
     * @return causal mask of shape [T × T]
     */
    public static float[][] causalMask(int seqLen) {
        float[][] mask = new float[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < seqLen; j++)
                mask[i][j] = (j > i) ? MASK_VALUE : 0.0f;
        return mask;
    }

    /**
     * Create a padding mask for encoder self-attention.
     *
     * <p>Positions where the token ID equals {@code padTokenId} are masked out.
     * Returns a [seqLen × seqLen] float matrix where:
     * <pre>
     *   mask[i][j] = -∞   if tokenIds[j] == padTokenId  (key position is padding)
     *   mask[i][j] = 0    otherwise
     * </pre>
     *
     * <p>All query positions i see the same mask (padding is determined by key position j).
     *
     * @param tokenIds  1-D array of token IDs for one sequence, length seqLen
     * @param padTokenId token ID that represents padding (typically 0)
     * @return padding mask of shape [seqLen × seqLen]
     */
    public static float[][] paddingMask(int[] tokenIds, int padTokenId) {
        int seqLen = tokenIds.length;
        float[][] mask = new float[seqLen][seqLen];
        for (int i = 0; i < seqLen; i++)
            for (int j = 0; j < seqLen; j++)
                mask[i][j] = (tokenIds[j] == padTokenId) ? MASK_VALUE : 0.0f;
        return mask;
    }

    /**
     * Create a cross-attention padding mask for decoder-encoder attention.
     *
     * <p>The decoder queries attend over encoder keys/values. Encoder padding positions
     * should not be attended to.
     *
     * @param encoderTokenIds token IDs for the encoder sequence, length srcLen
     * @param decoderSeqLen   length of the decoder sequence (tgtLen)
     * @param padTokenId      token ID that represents padding
     * @return cross-attention mask of shape [tgtLen × srcLen]
     */
    public static float[][] crossAttentionPaddingMask(int[] encoderTokenIds,
                                                       int decoderSeqLen,
                                                       int padTokenId) {
        int srcLen = encoderTokenIds.length;
        float[][] mask = new float[decoderSeqLen][srcLen];
        for (int i = 0; i < decoderSeqLen; i++)
            for (int j = 0; j < srcLen; j++)
                mask[i][j] = (encoderTokenIds[j] == padTokenId) ? MASK_VALUE : 0.0f;
        return mask;
    }

    /**
     * Combine two masks by element-wise addition.
     * For example, combine causal mask + padding mask for decoder self-attention.
     *
     * @param mask1 first mask [M × N]
     * @param mask2 second mask [M × N]
     * @return combined mask [M × N]
     */
    public static float[][] combineMasks(float[][] mask1, float[][] mask2) {
        int rows = mask1.length;
        int cols = mask1[0].length;
        if (mask2.length != rows || mask2[0].length != cols)
            throw new IllegalArgumentException(
                "combineMasks: shapes [" + rows + " × " + cols + "] and [" +
                mask2.length + " × " + mask2[0].length + "] do not match");

        float[][] result = new float[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                // Adding two -∞ values stays -∞; adding 0+0=0; adding 0+-∞=-∞
                float sum = mask1[r][c] + mask2[r][c];
                // Clamp: two NEGATIVE_INFINITY additions can produce NaN on some JVMs
                result[r][c] = Float.isNaN(sum) ? MASK_VALUE : sum;
            }
        return result;
    }

    /**
     * Print a mask matrix for debugging.
     */
    public static String formatMask(float[][] mask) {
        StringBuilder sb = new StringBuilder();
        sb.append("Mask [").append(mask.length).append(" × ").append(mask[0].length).append("]:\n");
        for (float[] row : mask) {
            sb.append("  [");
            for (int c = 0; c < row.length; c++) {
                if (row[c] == MASK_VALUE) sb.append("  -inf");
                else sb.append(String.format("  %4.1f", row[c]));
                if (c < row.length - 1) sb.append(",");
            }
            sb.append("]\n");
        }
        return sb.toString();
    }
}
