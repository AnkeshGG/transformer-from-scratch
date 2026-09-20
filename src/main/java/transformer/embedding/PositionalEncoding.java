/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.embedding;

import transformer.TransformerConfig;
import transformer.tensor.Tensor3D;

/**
 * PositionalEncoding — Sinusoidal positional encoding from the original Transformer paper.
 *
 * <h2>Motivation</h2>
 * <p>The Transformer's self-attention mechanism is permutation-equivariant: it treats
 * all positions identically and has no built-in notion of sequence order. To inject
 * positional information, a positional encoding is added to the token embeddings.
 *
 * <h2>Mathematical definition (Vaswani et al., 2017, Section 3.5)</h2>
 * <p>For position {@code pos} in the sequence and dimension index {@code i}:
 * <pre>
 *   PE(pos, 2i)   = sin(pos / 10000^(2i / dModel))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i / dModel))
 * </pre>
 *
 * <p>Equivalently, the denominator term is:
 * <pre>
 *   div_term_i = 10000^(2i / dModel)
 *              = exp(2i × log(10000) / dModel)
 * </pre>
 *
 * <p>This is computed using the log form for numerical stability.
 *
 * <h2>Intuition</h2>
 * <ul>
 *   <li>Even dimensions use sine, odd dimensions use cosine.
 *   <li>Low-frequency oscillations (large wavelengths) encode coarse position;
 *       high-frequency oscillations (small wavelengths) encode fine position.
 *   <li>For any fixed offset Δ, PE(pos + Δ) can be expressed as a linear function
 *       of PE(pos), which allows the model to attend by relative positions.
 * </ul>
 *
 * <h2>Dimensions</h2>
 * <pre>
 *   Input:  Tensor3D [B × S × dModel]  — token embeddings
 *   Output: Tensor3D [B × S × dModel]  — embeddings + positional encoding
 * </pre>
 *
 * <p>The PE table has shape [maxSequenceLength × dModel] and is precomputed once
 * in the constructor (it is not trainable in the original paper).
 */
public final class PositionalEncoding {

    /**
     * Precomputed positional encoding table.
     * Shape: [maxSequenceLength × dModel]
     * pe[pos][d] = sin or cos value for position pos, dimension d.
     */
    private final float[][] pe;

    /** Maximum supported sequence length. */
    private final int maxSequenceLength;

    /** Model embedding dimension. */
    private final int dModel;

    /** Debug mode flag. */
    private final boolean debug;

    // ── Constructor ──────────────────────────────────────────────────────────

    /**
     * Precompute the sinusoidal encoding table.
     *
     * @param config model configuration (uses maxSequenceLength and dModel)
     */
    public PositionalEncoding(TransformerConfig config) {
        this.maxSequenceLength = config.maxSequenceLength;
        this.dModel            = config.dModel;
        this.debug             = config.debugMode;
        this.pe                = computeTable(maxSequenceLength, dModel);
    }

    // ── Core computation ─────────────────────────────────────────────────────

    /**
     * Precompute the full PE table of shape [maxLen × dModel].
     *
     * <p>Algorithm:
     * <ol>
     *   <li>For each position pos in [0, maxLen):
     *   <li>  For each pair of dimensions i in [0, dModel/2):
     *   <li>    div_term = exp(2i × log(10000) / dModel)
     *   <li>    pe[pos][2i]   = sin(pos / div_term)
     *   <li>    pe[pos][2i+1] = cos(pos / div_term)
     * </ol>
     */
    private static float[][] computeTable(int maxLen, int dModel) {
        float[][] table = new float[maxLen][dModel];
        double logBase = Math.log(10000.0);

        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < dModel / 2; i++) {
                // div_term = 10000^(2i/dModel) = exp(2i * log(10000) / dModel)
                double divTerm = Math.exp((2.0 * i * logBase) / dModel);
                double angle   = pos / divTerm;

                table[pos][2 * i]     = (float) Math.sin(angle);   // even dim → sin
                // Guard: if dModel is odd, don't write past last column
                if (2 * i + 1 < dModel)
                    table[pos][2 * i + 1] = (float) Math.cos(angle);  // odd dim → cos
            }
        }
        return table;
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Add positional encoding to token embeddings.
     *
     * <p>Operation:
     * <pre>
     *   output[b][s][d] = input[b][s][d] + PE[s][d]
     * </pre>
     *
     * <p>The positional encoding is the same for every item in the batch
     * (it depends only on position, not on the token content).
     *
     * @param embeddings token embeddings, shape [B × S × dModel]
     * @return embeddings with positional encoding added, same shape [B × S × dModel]
     * @throws IllegalArgumentException if sequence length exceeds maxSequenceLength
     */
    public Tensor3D forward(Tensor3D embeddings) {
        int batchSize = embeddings.batch;
        int seqLen    = embeddings.rows;

        if (seqLen > maxSequenceLength)
            throw new IllegalArgumentException(
                "Sequence length " + seqLen + " exceeds maxSequenceLength " + maxSequenceLength);
        if (embeddings.cols != dModel)
            throw new IllegalArgumentException(
                "Embedding dimension " + embeddings.cols + " does not match dModel " + dModel);

        if (debug) {
            System.out.printf("[PositionalEncoding] input shape: [%d, %d, %d]%n",
                batchSize, seqLen, dModel);
        }

        Tensor3D output = new Tensor3D(batchSize, seqLen, dModel);

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int d = 0; d < dModel; d++) {
                    output.data[b][s][d] = embeddings.data[b][s][d] + pe[s][d];
                }
            }
        }

        if (debug) {
            System.out.printf("[PositionalEncoding] output shape: [%d, %d, %d]%n",
                batchSize, seqLen, dModel);
        }

        return output;
    }

    // ── Inspection utilities ─────────────────────────────────────────────────

    /**
     * Return the positional encoding for a specific position and dimension.
     * Useful for unit testing and visualisation.
     *
     * @param position sequence position (0-indexed)
     * @param dim      dimension index
     * @return PE value
     */
    public float getValue(int position, int dim) {
        return pe[position][dim];
    }

    /**
     * Return the full encoding for a given position as a copy.
     *
     * @param position sequence position (0-indexed)
     * @return float array of length dModel
     */
    public float[] getPositionEncoding(int position) {
        float[] result = new float[dModel];
        System.arraycopy(pe[position], 0, result, 0, dModel);
        return result;
    }
}
