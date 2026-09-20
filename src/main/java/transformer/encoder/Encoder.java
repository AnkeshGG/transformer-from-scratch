/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.encoder;

import transformer.TransformerConfig;
import transformer.attention.AttentionMask;
import transformer.embedding.PositionalEncoding;
import transformer.embedding.TokenEmbedding;
import transformer.tensor.Tensor3D;

import java.util.Random;

/**
 * Encoder — The complete encoder stack of the Transformer.
 *
 * <h2>Architecture</h2>
 * <p>The encoder consists of:
 * <ol>
 *   <li>Token embedding lookup (scaled by sqrt(dModel))
 *   <li>Sinusoidal positional encoding (added to embeddings)
 *   <li>N identical encoder layers stacked in sequence
 * </ol>
 *
 * <pre>
 *   Source tokens [B × S]
 *         ↓
 *   TokenEmbedding    [B × S × dModel]
 *         ↓
 *   + PositionalEncoding
 *         ↓
 *   EncoderLayer 1
 *         ↓
 *   EncoderLayer 2
 *         ↓
 *        ...
 *         ↓
 *   EncoderLayer N
 *         ↓
 *   Encoder Output  [B × S × dModel]
 * </pre>
 *
 * <h2>What the encoder produces</h2>
 * <p>The encoder transforms a sequence of discrete token IDs into a sequence of
 * continuous, context-aware representations. Each position's representation
 * encodes information about the entire source sequence (through bi-directional
 * self-attention).
 *
 * <p>These encoder hidden states are then used as keys and values in the
 * decoder's cross-attention layers.
 *
 * <h2>Dimensions</h2>
 * <pre>
 *   Input:  int[][] [B × S]        — batch of token ID sequences
 *   Output: Tensor3D [B × S × dModel] — encoder hidden states
 * </pre>
 */
public final class Encoder {

    /** Token embedding layer (vocabSize × dModel lookup table, scaled by sqrt(dModel)). */
    private final TokenEmbedding tokenEmbedding;

    /** Sinusoidal positional encoding (precomputed, not trainable). */
    private final PositionalEncoding positionalEncoding;

    /** Stack of N encoder layers. */
    private final EncoderLayer[] layers;

    /** Padding token ID (used to create padding mask). */
    private final int padTokenId;

    /** Debug mode flag. */
    private final boolean debug;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct the encoder stack.
     *
     * @param config    model configuration
     * @param padTokenId token ID for padding (typically 0)
     * @param rng       seeded random generator for all layer initialisations
     */
    public Encoder(TransformerConfig config, int padTokenId, Random rng) {
        this.tokenEmbedding    = new TokenEmbedding(config);
        this.positionalEncoding = new PositionalEncoding(config);
        this.padTokenId        = padTokenId;
        this.debug             = config.debugMode;

        // Create N encoder layers, sharing the same RNG (each layer draws
        // its own random weights in sequence → deterministic with fixed seed).
        this.layers = new EncoderLayer[config.numEncoderLayers];
        for (int i = 0; i < config.numEncoderLayers; i++) {
            layers[i] = new EncoderLayer(config, rng);
        }
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Encode a batch of token ID sequences.
     *
     * <p>Pipeline:
     * <ol>
     *   <li>Look up token embeddings (scaled by sqrt(dModel)).
     *   <li>Add sinusoidal positional encoding.
     *   <li>Pass through each of the N encoder layers.
     * </ol>
     *
     * <p>Padding tokens are masked so they do not contribute to attention.
     * This is done via a per-batch-item padding mask.
     *
     * @param tokenIds source token IDs [B × S]
     * @return encoder hidden states [B × S × dModel]
     */
    public Tensor3D forward(int[][] tokenIds) {
        int batchSize = tokenIds.length;
        int seqLen    = tokenIds[0].length;

        if (debug) {
            System.out.printf("[Encoder] Input token IDs: [%d × %d]%n", batchSize, seqLen);
        }

        // ── Step 1: Token embedding ────────────────────────────────────
        Tensor3D x = tokenEmbedding.forward(tokenIds);    // [B × S × dModel]

        // ── Step 2: Add positional encoding ───────────────────────────
        x = positionalEncoding.forward(x);                // [B × S × dModel]

        if (debug) {
            System.out.printf("[Encoder] After embedding+PE: [%d,%d,%d]%n",
                x.batch, x.rows, x.cols);
        }

        // ── Step 3: Build padding mask ─────────────────────────────────
        // Build padding mask from the first batch item's token IDs [S × S].
        // TODO: extend to per-item masks for heterogeneous batches.
        float[][] paddingMask = null;
        if (batchSize > 0) {
            paddingMask = AttentionMask.paddingMask(tokenIds[0], padTokenId);
        }

        // ── Step 4: Pass through N encoder layers ─────────────────────
        for (int layerIdx = 0; layerIdx < layers.length; layerIdx++) {
            if (debug) System.out.printf("[Encoder] Layer %d%n", layerIdx + 1);
            x = layers[layerIdx].forward(x, paddingMask);
        }

        if (debug) {
            System.out.printf("[Encoder] Output: [%d,%d,%d]%n", x.batch, x.rows, x.cols);
        }

        return x;  // [B × S × dModel]
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /**
     * Return the token embedding layer (useful for weight tying with output projection).
     */
    public TokenEmbedding getTokenEmbedding() {
        return tokenEmbedding;
    }

    /**
     * Return the number of encoder layers (N).
     */
    public int getNumLayers() {
        return layers.length;
    }
}
