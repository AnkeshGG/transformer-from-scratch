/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.encoder;

import transformer.TransformerConfig;
import transformer.attention.AttentionMask;
import transformer.attention.MultiHeadAttention;
import transformer.layers.FeedForward;
import transformer.layers.LayerNorm;
import transformer.tensor.Tensor3D;

import java.util.Random;

/**
 * EncoderLayer — A single encoder layer from the Transformer architecture.
 *
 * <h2>Architecture (Post-LN, original paper)</h2>
 * <p>Each encoder layer contains two sub-layers with residual connections and
 * layer normalisation applied <em>after</em> each sub-layer:
 *
 * <pre>
 *   Input x (shape: [B × S × dModel])
 *      │
 *      ├──────────────────────┐
 *      ↓                      │  (residual)
 *   MultiHeadSelfAttention    │
 *      ↓                      │
 *   Add ──────────────────────┘
 *      ↓
 *   LayerNorm
 *      │ (call this x₁)
 *      ├──────────────────────┐
 *      ↓                      │  (residual)
 *   FeedForward               │
 *      ↓                      │
 *   Add ──────────────────────┘
 *      ↓
 *   LayerNorm
 *      ↓
 *   Output (shape: [B × S × dModel])
 * </pre>
 *
 * <p>Formally:
 * <pre>
 *   x₁ = LayerNorm(x + MultiHeadSelfAttention(x, x, x))
 *   x₂ = LayerNorm(x₁ + FeedForward(x₁))
 * </pre>
 *
 * <h2>Key facts about the encoder</h2>
 * <ul>
 *   <li>Self-attention: queries, keys, and values all come from the same input x.
 *   <li>No causal mask: the encoder can attend to all positions in both directions.
 *   <li>Optional padding mask: prevents attending to PAD tokens.
 *   <li>Each position's representation depends on all other positions after attention.
 * </ul>
 *
 * <h2>Dimensions</h2>
 * <pre>
 *   Input:  [B × S × dModel]
 *   Output: [B × S × dModel]  (same shape — residuals preserve dimensions)
 * </pre>
 */
public final class EncoderLayer {

    /** Multi-Head Self-Attention sub-layer. */
    private final MultiHeadAttention selfAttention;

    /** Layer normalisation after the attention sub-layer. */
    private final LayerNorm norm1;

    /** Position-wise Feed-Forward sub-layer. */
    private final FeedForward feedForward;

    /** Layer normalisation after the feed-forward sub-layer. */
    private final LayerNorm norm2;

    /** Debug mode flag. */
    private final boolean debug;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct an EncoderLayer.
     *
     * @param config model configuration
     * @param rng    seeded random generator for weight initialisation
     */
    public EncoderLayer(TransformerConfig config, Random rng) {
        this.selfAttention = new MultiHeadAttention(config, rng);
        this.norm1         = new LayerNorm(config.dModel, config.layerNormEpsilon);
        this.feedForward   = new FeedForward(config.dModel, config.dFF, rng);
        this.norm2         = new LayerNorm(config.dModel, config.layerNormEpsilon);
        this.debug         = config.debugMode;
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Apply one encoder layer to the input.
     *
     * <p>The encoder does NOT use a causal mask — all positions can attend to
     * all other positions. An optional padding mask can be provided to prevent
     * attending to padding tokens.
     *
     * @param x           input tensor [B × S × dModel]
     * @param paddingMask optional additive mask [S × S]; null = no masking.
     *                    Use {@link AttentionMask#paddingMask} to create one.
     * @return output tensor [B × S × dModel]
     */
    public Tensor3D forward(Tensor3D x, float[][] paddingMask) {

        if (debug) {
            System.out.printf("[EncoderLayer] Input: [%d,%d,%d]%n",
                x.batch, x.rows, x.cols);
        }

        // ── Sub-layer 1: Multi-Head Self-Attention ─────────────────────
        // Compute: attnOutput = MultiHeadSelfAttention(x, x, x, paddingMask)
        // Residual + Norm: x₁ = LayerNorm(x + attnOutput)
        Tensor3D attnOutput = selfAttention.selfAttention(x, paddingMask);
        Tensor3D afterAdd1  = x.add(attnOutput);        // residual connection
        Tensor3D x1         = norm1.forward(afterAdd1); // layer norm

        if (debug) {
            System.out.printf("[EncoderLayer] After Self-Attn+Norm: [%d,%d,%d]%n",
                x1.batch, x1.rows, x1.cols);
        }

        // ── Sub-layer 2: Position-wise Feed-Forward ────────────────────
        // Compute: ffnOutput = FeedForward(x₁)
        // Residual + Norm: x₂ = LayerNorm(x₁ + ffnOutput)
        Tensor3D ffnOutput  = feedForward.forward(x1);
        Tensor3D afterAdd2  = x1.add(ffnOutput);        // residual connection
        Tensor3D x2         = norm2.forward(afterAdd2); // layer norm

        if (debug) {
            System.out.printf("[EncoderLayer] After FFN+Norm: [%d,%d,%d]%n",
                x2.batch, x2.rows, x2.cols);
        }

        return x2;
    }
}
