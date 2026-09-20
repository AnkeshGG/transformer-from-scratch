/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.decoder;

import transformer.TransformerConfig;
import transformer.attention.AttentionMask;
import transformer.attention.MultiHeadAttention;
import transformer.layers.FeedForward;
import transformer.layers.LayerNorm;
import transformer.tensor.Tensor3D;

import java.util.Random;

/**
 * DecoderLayer — A single decoder layer from the Transformer architecture.
 *
 * <h2>Architecture (Post-LN, original paper)</h2>
 * <p>Each decoder layer contains THREE sub-layers:
 *
 * <pre>
 *   Decoder input x (shape: [B × T × dModel])
 *         │
 *         ├──────────────────────────────┐
 *         ↓                              │  (residual)
 *   Masked MultiHeadSelfAttention        │  ← causal mask applied
 *         ↓                              │
 *   Add ─────────────────────────────────┘
 *         ↓
 *   LayerNorm
 *         │ (call this x₁)
 *         │
 *         ├──────────────────────────────┐
 *         ↓                              │  (residual)
 *   MultiHeadCrossAttention              │  ← Q from x₁, K/V from encoder output
 *         ↓                              │
 *   Add ─────────────────────────────────┘
 *         ↓
 *   LayerNorm
 *         │ (call this x₂)
 *         │
 *         ├──────────────────────────────┐
 *         ↓                              │  (residual)
 *   FeedForward                          │
 *         ↓                              │
 *   Add ─────────────────────────────────┘
 *         ↓
 *   LayerNorm
 *         ↓
 *   Output (shape: [B × T × dModel])
 * </pre>
 *
 * <p>Formally:
 * <pre>
 *   x₁ = LayerNorm(x  + MaskedSelfAttention(x, x, x, causal_mask))
 *   x₂ = LayerNorm(x₁ + CrossAttention(x₁, encoderOutput, encoderOutput))
 *   x₃ = LayerNorm(x₂ + FeedForward(x₂))
 * </pre>
 *
 * <h2>Why three sub-layers?</h2>
 * <ul>
 *   <li><b>Masked Self-Attention</b>: Allows each decoder position to attend to
 *       past positions in the target (auto-regressive generation). Future positions
 *       are masked with −∞ to prevent information leakage.
 *   <li><b>Cross-Attention</b>: Allows the decoder to attend over the encoder's
 *       output representations, connecting the encoder and decoder.
 *   <li><b>Feed-Forward</b>: Position-wise transformation applied after attention.
 * </ul>
 *
 * <h2>Causal mask in detail</h2>
 * <p>During training with teacher forcing, all target positions are available.
 * The causal mask ensures position i can only see positions 0..i:
 * <pre>
 *   For T=4:
 *     [0,  −∞,  −∞,  −∞]   ← position 0 sees only itself
 *     [0,   0,  −∞,  −∞]   ← position 1 sees positions 0,1
 *     [0,   0,   0,  −∞]   ← position 2 sees positions 0,1,2
 *     [0,   0,   0,   0]   ← position 3 sees all
 * </pre>
 *
 * <h2>Cross-attention keys/values</h2>
 * <p>In cross-attention:
 * <ul>
 *   <li>Q comes from the decoder's current hidden state x₁  [B × T × dModel]
 *   <li>K, V come from the encoder's output                  [B × S × dModel]
 *   <li>The attention scores matrix has shape [T × S] (decoder positions × encoder positions)
 * </ul>
 *
 * <h2>Dimensions</h2>
 * <pre>
 *   decoderInput:   [B × T × dModel]
 *   encoderOutput:  [B × S × dModel]
 *   Output:         [B × T × dModel]
 * </pre>
 */
public final class DecoderLayer {

    // ── Sub-layers ───────────────────────────────────────────────────────────

    /** Sub-layer 1: Masked Multi-Head Self-Attention (uses causal mask). */
    private final MultiHeadAttention maskedSelfAttention;

    /** Layer norm after sub-layer 1. */
    private final LayerNorm norm1;

    /** Sub-layer 2: Multi-Head Cross-Attention (Q from decoder, K/V from encoder). */
    private final MultiHeadAttention crossAttention;

    /** Layer norm after sub-layer 2. */
    private final LayerNorm norm2;

    /** Sub-layer 3: Position-wise Feed-Forward Network. */
    private final FeedForward feedForward;

    /** Layer norm after sub-layer 3. */
    private final LayerNorm norm3;

    /** Debug mode flag. */
    private final boolean debug;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct a DecoderLayer.
     *
     * @param config model configuration
     * @param rng    seeded random generator for weight initialisation
     */
    public DecoderLayer(TransformerConfig config, Random rng) {
        this.maskedSelfAttention = new MultiHeadAttention(config, rng);
        this.norm1               = new LayerNorm(config.dModel, config.layerNormEpsilon);
        this.crossAttention      = new MultiHeadAttention(config, rng);
        this.norm2               = new LayerNorm(config.dModel, config.layerNormEpsilon);
        this.feedForward         = new FeedForward(config.dModel, config.dFF, rng);
        this.norm3               = new LayerNorm(config.dModel, config.layerNormEpsilon);
        this.debug               = config.debugMode;
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Apply one decoder layer.
     *
     * @param x             decoder input (target embeddings + PE) [B × T × dModel]
     * @param encoderOutput encoder hidden states [B × S × dModel]
     * @param causalMask    causal mask [T × T] (use {@link AttentionMask#causalMask})
     * @param crossMask     optional cross-attention padding mask [T × S]; null = no masking
     * @return decoder layer output [B × T × dModel]
     */
    public Tensor3D forward(Tensor3D x, Tensor3D encoderOutput,
                             float[][] causalMask, float[][] crossMask) {

        if (debug) {
            System.out.printf("[DecoderLayer] Input: [%d,%d,%d], Encoder: [%d,%d,%d]%n",
                x.batch, x.rows, x.cols,
                encoderOutput.batch, encoderOutput.rows, encoderOutput.cols);
        }

        // ── Sub-layer 1: Masked Self-Attention ────────────────────────
        // Q, K, V all come from x (self-attention).
        // Causal mask prevents attending to future positions.
        //
        // selfAttnOut = MaskedSelfAttention(x, x, x, causal_mask)
        // x₁ = LayerNorm(x + selfAttnOut)
        Tensor3D selfAttnOut = maskedSelfAttention.selfAttention(x, causalMask);
        Tensor3D x1          = norm1.forward(x.add(selfAttnOut));

        if (debug) {
            System.out.printf("[DecoderLayer] After Masked Self-Attn+Norm: [%d,%d,%d]%n",
                x1.batch, x1.rows, x1.cols);
        }

        // ── Sub-layer 2: Cross-Attention (Encoder-Decoder Attention) ──
        // Q comes from x₁ (decoder states).
        // K, V come from encoderOutput (encoder hidden states).
        //
        // This is the mechanism that connects encoder and decoder:
        // each decoder position can attend to all encoder positions.
        //
        // crossAttnOut = CrossAttention(Q=x₁, KV=encoderOutput, crossMask)
        // x₂ = LayerNorm(x₁ + crossAttnOut)
        Tensor3D crossAttnOut = crossAttention.crossAttention(x1, encoderOutput, crossMask);
        Tensor3D x2           = norm2.forward(x1.add(crossAttnOut));

        if (debug) {
            System.out.printf("[DecoderLayer] After Cross-Attn+Norm: [%d,%d,%d]%n",
                x2.batch, x2.rows, x2.cols);
        }

        // ── Sub-layer 3: Feed-Forward Network ─────────────────────────
        // ffnOut = FeedForward(x₂)
        // x₃ = LayerNorm(x₂ + ffnOut)
        Tensor3D ffnOut = feedForward.forward(x2);
        Tensor3D x3     = norm3.forward(x2.add(ffnOut));

        if (debug) {
            System.out.printf("[DecoderLayer] After FFN+Norm: [%d,%d,%d]%n",
                x3.batch, x3.rows, x3.cols);
        }

        return x3;  // [B × T × dModel]
    }
}
