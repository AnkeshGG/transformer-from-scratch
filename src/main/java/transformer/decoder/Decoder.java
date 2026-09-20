/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.decoder;

import transformer.TransformerConfig;
import transformer.attention.AttentionMask;
import transformer.embedding.PositionalEncoding;
import transformer.embedding.TokenEmbedding;
import transformer.tensor.Tensor3D;

import java.util.Random;

/**
 * Decoder — The complete decoder stack of the Transformer.
 *
 * <h2>Architecture</h2>
 * <p>The decoder consists of:
 * <ol>
 *   <li>Target token embedding lookup (scaled by sqrt(dModel))
 *   <li>Sinusoidal positional encoding (added to target embeddings)
 *   <li>N identical decoder layers stacked in sequence
 * </ol>
 *
 * <pre>
 *   Target tokens [B × T]                  Encoder output [B × S × dModel]
 *         ↓                                         ↓
 *   TokenEmbedding    [B × T × dModel]              │
 *         ↓                                         │
 *   + PositionalEncoding                            │
 *         ↓                                         │
 *   DecoderLayer 1 ←─── cross-attention ────────────┤
 *         ↓                                         │
 *   DecoderLayer 2 ←─── cross-attention ────────────┤
 *         ↓                                         │
 *        ...                                        │
 *         ↓                                         │
 *   DecoderLayer N ←─── cross-attention ────────────┘
 *         ↓
 *   Decoder Output [B × T × dModel]
 * </pre>
 *
 * <h2>Teacher forcing (training)</h2>
 * <p>During training, the entire target sequence (shifted right by one position)
 * is fed as input simultaneously, with a causal mask to prevent position i from
 * seeing position j > i. This is called "teacher forcing" and is more efficient
 * than autoregressive generation during training.
 *
 * <h2>Autoregressive generation (inference)</h2>
 * <p>During inference, the decoder generates one token at a time. Starting with
 * BOS, each new token is appended and the decoder is re-run. This is implemented
 * in {@link transformer.Transformer#generate}.
 *
 * <h2>Dimensions</h2>
 * <pre>
 *   Input (target):  int[][] [B × T]          — batch of target token ID sequences
 *   encoderOutput:   Tensor3D [B × S × dModel] — encoder hidden states
 *   Output:          Tensor3D [B × T × dModel] — decoder hidden states
 * </pre>
 */
public final class Decoder {

    /** Target token embedding layer. */
    private final TokenEmbedding tokenEmbedding;

    /** Sinusoidal positional encoding for target positions. */
    private final PositionalEncoding positionalEncoding;

    /** Stack of N decoder layers. */
    private final DecoderLayer[] layers;

    /** Padding token ID. */
    private final int padTokenId;

    /** Debug mode flag. */
    private final boolean debug;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct the decoder stack.
     *
     * @param config     model configuration
     * @param padTokenId padding token ID
     * @param rng        seeded random generator for all layer weights
     */
    public Decoder(TransformerConfig config, int padTokenId, Random rng) {
        this.tokenEmbedding     = new TokenEmbedding(config);
        this.positionalEncoding = new PositionalEncoding(config);
        this.padTokenId         = padTokenId;
        this.debug              = config.debugMode;

        this.layers = new DecoderLayer[config.numDecoderLayers];
        for (int i = 0; i < config.numDecoderLayers; i++) {
            layers[i] = new DecoderLayer(config, rng);
        }
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Decode a batch of target token sequences given encoder hidden states.
     *
     * <p>Pipeline:
     * <ol>
     *   <li>Embed target tokens (scaled by sqrt(dModel)).
     *   <li>Add positional encoding.
     *   <li>Build causal mask (prevents future token leakage).
     *   <li>Optionally build cross-attention padding mask.
     *   <li>Pass through each of the N decoder layers.
     * </ol>
     *
     * @param targetTokenIds target token IDs [B × T]
     *                       (shifted-right during training: starts with BOS)
     * @param encoderOutput  encoder hidden states [B × S × dModel]
     * @param encoderTokenIds source token IDs [B × S] (used for cross-attention padding mask)
     * @return decoder hidden states [B × T × dModel]
     */
    public Tensor3D forward(int[][] targetTokenIds, Tensor3D encoderOutput,
                             int[][] encoderTokenIds) {

        int batchSize = targetTokenIds.length;
        int tgtLen    = targetTokenIds[0].length;
        int srcLen    = encoderOutput.rows;

        if (debug) {
            System.out.printf("[Decoder] Target tokens: [%d × %d], Encoder output: [%d,%d,%d]%n",
                batchSize, tgtLen, encoderOutput.batch, encoderOutput.rows, encoderOutput.cols);
        }

        // ── Step 1: Target token embedding ────────────────────────────
        Tensor3D x = tokenEmbedding.forward(targetTokenIds);   // [B × T × dModel]

        // ── Step 2: Add positional encoding ───────────────────────────
        x = positionalEncoding.forward(x);                      // [B × T × dModel]

        if (debug) {
            System.out.printf("[Decoder] After target embedding+PE: [%d,%d,%d]%n",
                x.batch, x.rows, x.cols);
        }

        // ── Step 3: Build causal mask [T × T] ─────────────────────────
        // Position i cannot attend to j > i.
        float[][] causalMask = AttentionMask.causalMask(tgtLen);

        // ── Step 4: Build cross-attention padding mask [T × S] ────────
        // Prevents decoder from attending to encoder PAD positions.
        float[][] crossMask = null;
        if (encoderTokenIds != null && encoderTokenIds.length > 0) {
            crossMask = AttentionMask.crossAttentionPaddingMask(
                encoderTokenIds[0], tgtLen, padTokenId);
        }

        // ── Step 5: Pass through N decoder layers ─────────────────────
        for (int layerIdx = 0; layerIdx < layers.length; layerIdx++) {
            if (debug) System.out.printf("[Decoder] Layer %d%n", layerIdx + 1);
            x = layers[layerIdx].forward(x, encoderOutput, causalMask, crossMask);
        }

        if (debug) {
            System.out.printf("[Decoder] Output: [%d,%d,%d]%n", x.batch, x.rows, x.cols);
        }

        return x;  // [B × T × dModel]
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /**
     * Return the target token embedding (useful for weight tying).
     */
    public TokenEmbedding getTokenEmbedding() {
        return tokenEmbedding;
    }

    /**
     * Return the number of decoder layers (N).
     */
    public int getNumLayers() {
        return layers.length;
    }
}
