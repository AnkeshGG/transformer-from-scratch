/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.attention;

import transformer.tensor.Matrix;

/**
 * ScaledDotProductAttention — the core attention mechanism.
 *
 * <h2>Mathematical operation</h2>
 * <p>Given query Q, key K, value V matrices:
 * <pre>
 *   Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) + mask ) · V
 *
 *   where:
 *     Q ∈ R^{S_q × d_k}   — queries  (S_q query positions)
 *     K ∈ R^{S_k × d_k}   — keys     (S_k key/value positions)
 *     V ∈ R^{S_k × d_v}   — values   (same S_k positions as K)
 *     d_k                  — query/key dimension (dHead in multi-head attention)
 *
 *   QK^T ∈ R^{S_q × S_k}  — raw attention scores
 *   / sqrt(d_k)            — scaling to prevent dot-products from growing large
 *   + mask                 — optional additive mask (0 or -∞) applied before softmax
 *   softmax(·)             — row-wise softmax → attention weights in [0,1] summing to 1
 *   · V                    — weighted sum of values
 *   output ∈ R^{S_q × d_v} — attended context vectors
 * </pre>
 *
 * <h2>Why scale by sqrt(d_k)?</h2>
 * <p>Without scaling, the dot products QK^T grow in magnitude as d_k increases
 * (variance of the dot product grows linearly with d_k). Large dot products push
 * the softmax into regions with very small gradients (saturation), which harms
 * training. Dividing by sqrt(d_k) normalises the variance to ~1.
 *
 * <h2>Why numerically stable softmax?</h2>
 * <p>Standard softmax exp(x) / Σ exp(x_j) overflows when x is large (e.g. x > 700).
 * The stable version subtracts the row maximum first:
 * <pre>
 *   softmax(x)_i = exp(x_i − max(x)) / Σ_j exp(x_j − max(x))
 * </pre>
 * This is mathematically identical but numerically safe.
 *
 * <h2>Cross-attention vs self-attention</h2>
 * <p>This class handles both:
 * <ul>
 *   <li><b>Self-attention</b>: Q = K = V come from the same sequence (S_q = S_k).
 *   <li><b>Cross-attention</b>: Q comes from the decoder (S_q = T),
 *       K and V come from the encoder (S_k = S). S_q ≠ S_k is allowed.
 * </ul>
 *
 * <h2>Dimension summary</h2>
 * <pre>
 *   Q:       [S_q × d_k]
 *   K:       [S_k × d_k]
 *   V:       [S_k × d_v]
 *   scores:  [S_q × S_k]   = Q × K^T / sqrt(d_k) + mask
 *   weights: [S_q × S_k]   = softmax(scores)
 *   output:  [S_q × d_v]   = weights × V
 * </pre>
 */
public final class ScaledDotProductAttention {

    // This class is stateless (no parameters); all state is in the calling layer.
    // It is instantiated once per head in MultiHeadAttention.
    private ScaledDotProductAttention() {}

    /**
     * Compute scaled dot-product attention.
     *
     * @param Q    query matrix [S_q × d_k]
     * @param K    key matrix   [S_k × d_k]
     * @param V    value matrix [S_k × d_v]
     * @param mask optional additive mask [S_q × S_k]; null means no masking.
     *             Masked positions should contain {@link AttentionMask#MASK_VALUE} (= −∞).
     * @return attention output [S_q × d_v]
     */
    public static Matrix compute(Matrix Q, Matrix K, Matrix V, float[][] mask) {

        // ── Validation ────────────────────────────────────────────────────
        if (Q.cols != K.cols)
            throw new IllegalArgumentException(
                "ScaledDotProductAttention: Q.cols (" + Q.cols +
                ") must equal K.cols (" + K.cols + ") [both are d_k].\n" +
                "  Q = " + Q.shapeString() + ", K = " + K.shapeString());
        if (K.rows != V.rows)
            throw new IllegalArgumentException(
                "ScaledDotProductAttention: K.rows (" + K.rows +
                ") must equal V.rows (" + V.rows + ") [both are S_k].\n" +
                "  K = " + K.shapeString() + ", V = " + V.shapeString());

        int sqLen = Q.rows;   // number of query positions
        int skLen = K.rows;   // number of key/value positions
        int dk    = Q.cols;   // key/query dimension

        // ── Step 1: Compute raw attention scores ────────────────────────
        // scores = Q × K^T / sqrt(d_k)
        //
        // Q      : [S_q × d_k]
        // K^T    : [d_k × S_k]
        // Q×K^T  : [S_q × S_k]
        //
        float scaleFactor = (float) (1.0 / Math.sqrt(dk));
        Matrix KT     = K.transpose();            // [d_k × S_k]
        Matrix scores = Q.matmul(KT).scale(scaleFactor); // [S_q × S_k]

        // ── Step 2: Apply mask (if provided) ────────────────────────────
        // The mask is an additive matrix of 0s and −∞s.
        // Adding −∞ to a score before softmax causes softmax to output ≈ 0 there.
        if (mask != null) {
            if (mask.length != sqLen || mask[0].length != skLen)
                throw new IllegalArgumentException(
                    "ScaledDotProductAttention: mask shape [" + mask.length +
                    " × " + mask[0].length + "] does not match scores shape [" +
                    sqLen + " × " + skLen + "]");

            // Add mask in-place by building a Matrix from the mask values.
            float[][] maskedData = new float[sqLen][skLen];
            for (int i = 0; i < sqLen; i++)
                for (int j = 0; j < skLen; j++)
                    maskedData[i][j] = scores.data[i][j] + mask[i][j];
            scores = new Matrix(maskedData);
        }

        // ── Step 3: Row-wise stable softmax ────────────────────────────
        // weights[i][j] = probability that query i attends to key j
        // Each row sums to 1.
        //
        // weights : [S_q × S_k]
        Matrix weights = scores.softmax();

        // ── Step 4: Weighted sum of values ──────────────────────────────
        // output = weights × V
        //
        // weights : [S_q × S_k]
        // V       : [S_k × d_v]
        // output  : [S_q × d_v]
        //
        return weights.matmul(V);
    }

    /**
     * Compute scaled dot-product attention and also return the attention weights.
     * Useful for visualising which positions the model attends to.
     *
     * @param Q    query matrix [S_q × d_k]
     * @param K    key matrix   [S_k × d_k]
     * @param V    value matrix [S_k × d_v]
     * @param mask optional additive mask [S_q × S_k]
     * @return AttentionResult containing both the output and the attention weights
     */
    public static AttentionResult computeWithWeights(Matrix Q, Matrix K, Matrix V, float[][] mask) {
        // Reuse the same logic but also capture weights.
        int dk = Q.cols;
        float scaleFactor = (float) (1.0 / Math.sqrt(dk));

        Matrix KT     = K.transpose();
        Matrix scores = Q.matmul(KT).scale(scaleFactor);

        if (mask != null) {
            float[][] maskedData = new float[Q.rows][K.rows];
            for (int i = 0; i < Q.rows; i++)
                for (int j = 0; j < K.rows; j++)
                    maskedData[i][j] = scores.data[i][j] + mask[i][j];
            scores = new Matrix(maskedData);
        }

        Matrix weights = scores.softmax();   // [S_q × S_k]
        Matrix output  = weights.matmul(V); // [S_q × d_v]

        return new AttentionResult(output, weights);
    }

    /**
     * Holds both the attention output and the attention weight matrix.
     * The weight matrix is useful for visualisation and debugging.
     */
    public record AttentionResult(
        /** Attended context vectors: [S_q × d_v]. */
        Matrix output,
        /**
         * Attention weights (post-softmax): [S_q × S_k].
         * weights[i][j] = how much query position i attends to key position j.
         */
        Matrix weights
    ) {}
}
