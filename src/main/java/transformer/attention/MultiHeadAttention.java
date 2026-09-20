/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.attention;

import transformer.TransformerConfig;
import transformer.layers.Linear;
import transformer.tensor.Matrix;
import transformer.tensor.Tensor3D;

import java.util.Random;

/**
 * MultiHeadAttention — Multi-Head Attention as defined in "Attention Is All You Need".
 *
 * <h2>Motivation</h2>
 * <p>Instead of performing a single attention function with d_model-dimensional
 * queries, keys, and values, it is beneficial to linearly project the queries,
 * keys, and values h times to d_head dimensions using different learned projections,
 * compute attention in parallel on each "head", then concatenate and project again.
 *
 * <p>This allows the model to jointly attend to information from different
 * representation subspaces at different positions.
 *
 * <h2>Mathematical operation</h2>
 * <pre>
 *   Given input X (for self-attention: Q_in = K_in = V_in = X):
 *
 *   Q = X_q · W_Q^T    [S_q × dModel] → [S_q × dModel]
 *   K = X_k · W_K^T    [S_k × dModel] → [S_k × dModel]
 *   V = X_v · W_V^T    [S_k × dModel] → [S_k × dModel]
 *
 *   Split into h heads along the dModel dimension:
 *     Q_i = Q[:, i·dHead : (i+1)·dHead]   [S_q × dHead]
 *     K_i = K[:, i·dHead : (i+1)·dHead]   [S_k × dHead]
 *     V_i = V[:, i·dHead : (i+1)·dHead]   [S_k × dHead]
 *
 *   For each head i:
 *     head_i = Attention(Q_i, K_i, V_i)   [S_q × dHead]
 *
 *   Concatenate heads:
 *     MultiHead = Concat(head_1, ..., head_h)   [S_q × dModel]
 *
 *   Apply output projection:
 *     output = MultiHead · W_O^T   [S_q × dModel]
 * </pre>
 *
 * <h2>Self-attention vs cross-attention</h2>
 * <ul>
 *   <li><b>Self-attention</b>: Q_in = K_in = V_in = X (same tensor).
 *       Used in encoder and decoder.
 *   <li><b>Cross-attention</b>: Q_in from decoder, K_in = V_in from encoder.
 *       Used only in decoder.
 * </ul>
 *
 * <h2>Head splitting</h2>
 * <p>We split dModel into h equal chunks of size dHead = dModel / h.
 * The tensor reshape approach:
 * <pre>
 *   [B × S × dModel]
 *   → [B × S × h × dHead]   (conceptually)
 *   → [B × h × S × dHead]   (transpose head and sequence dims)
 *
 *   In our implementation we represent this as:
 *   → h separate slices, each [B × S × dHead]
 *
 *   where each slice is one "head".
 * </pre>
 *
 * <h2>Dimensions (batched)</h2>
 * <pre>
 *   Input queryInput: [B × S_q × dModel]
 *   Input keyInput:   [B × S_k × dModel]
 *   Input valueInput: [B × S_k × dModel]
 *
 *   After WQ, WK, WV projection:
 *     Q: [B × S_q × dModel]
 *     K: [B × S_k × dModel]
 *     V: [B × S_k × dModel]
 *
 *   Per head (for head i):
 *     Q_i: [B × S_q × dHead]
 *     K_i: [B × S_k × dHead]
 *     V_i: [B × S_k × dHead]
 *     head_i output: [B × S_q × dHead]
 *
 *   After concatenation:
 *     [B × S_q × dModel]
 *
 *   After WO projection:
 *     [B × S_q × dModel]
 * </pre>
 */
public final class MultiHeadAttention {

    // ── Projection matrices ──────────────────────────────────────────────────
    // W_Q: projects queries   [dModel → dModel]
    // W_K: projects keys      [dModel → dModel]
    // W_V: projects values    [dModel → dModel]
    // W_O: output projection  [dModel → dModel]

    /** Query projection: W_Q ∈ R^{dModel × dModel} */
    private final Linear wQ;

    /** Key projection: W_K ∈ R^{dModel × dModel} */
    private final Linear wK;

    /** Value projection: W_V ∈ R^{dModel × dModel} */
    private final Linear wV;

    /** Output projection: W_O ∈ R^{dModel × dModel} */
    private final Linear wO;

    // ── Configuration ────────────────────────────────────────────────────────

    /** Total model dimension. */
    private final int dModel;

    /** Number of attention heads. */
    private final int numHeads;

    /**
     * Per-head dimension: dHead = dModel / numHeads.
     * This is d_k (and d_v) in the paper.
     */
    private final int dHead;

    /** Debug mode flag. */
    private final boolean debug;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct a MultiHeadAttention layer.
     *
     * @param config model configuration (provides dModel, numHeads, dHead)
     * @param rng    seeded random generator for weight initialisation
     */
    public MultiHeadAttention(TransformerConfig config, Random rng) {
        this.dModel   = config.dModel;
        this.numHeads = config.numHeads;
        this.dHead    = config.dHead;
        this.debug    = config.debugMode;

        // Assertion: dModel must be divisible by numHeads (checked in TransformerConfig).
        // But we add an explicit guard here as well for safety.
        if (dModel % numHeads != 0)
            throw new IllegalArgumentException(
                "MultiHeadAttention: dModel (" + dModel +
                ") must be divisible by numHeads (" + numHeads + ")");

        // All four projection matrices are dModel × dModel.
        // Each head will see a dHead-wide slice of the full projection.
        this.wQ = new Linear(dModel, dModel, rng);
        this.wK = new Linear(dModel, dModel, rng);
        this.wV = new Linear(dModel, dModel, rng);
        this.wO = new Linear(dModel, dModel, rng);
    }

    // ── Forward: Self-attention ───────────────────────────────────────────────

    /**
     * Self-attention forward pass: Q, K, V all come from the same input.
     *
     * @param x    input tensor [B × S × dModel]
     * @param mask optional attention mask [S × S]; null = no masking
     * @return output tensor [B × S × dModel]
     */
    public Tensor3D selfAttention(Tensor3D x, float[][] mask) {
        return forward(x, x, x, mask);
    }

    // ── Forward: Cross-attention ──────────────────────────────────────────────

    /**
     * Cross-attention forward pass: Q from decoder, K and V from encoder.
     *
     * <pre>
     *   Q = decoderHidden · W_Q^T    [B × T × dModel]
     *   K = encoderOutput · W_K^T    [B × S × dModel]
     *   V = encoderOutput · W_V^T    [B × S × dModel]
     * </pre>
     *
     * @param queryInput  decoder hidden states [B × T × dModel]
     * @param keyValue    encoder output [B × S × dModel]
     * @param mask        optional mask [T × S]; null = no masking
     * @return output tensor [B × T × dModel]
     */
    public Tensor3D crossAttention(Tensor3D queryInput, Tensor3D keyValue, float[][] mask) {
        return forward(queryInput, keyValue, keyValue, mask);
    }

    // ── Core forward implementation ───────────────────────────────────────────

    /**
     * General multi-head attention forward pass.
     *
     * <p>Step-by-step:
     * <ol>
     *   <li>Project Q, K, V through learned linear projections.
     *   <li>Split each into h heads along the dModel dimension.
     *   <li>Apply scaled dot-product attention to each head.
     *   <li>Concatenate head outputs.
     *   <li>Apply output projection W_O.
     * </ol>
     *
     * @param queryInput  queries source [B × S_q × dModel]
     * @param keyInput    keys source    [B × S_k × dModel]
     * @param valueInput  values source  [B × S_k × dModel]
     * @param mask        optional additive mask [S_q × S_k]; null = no masking
     * @return output [B × S_q × dModel]
     */
    public Tensor3D forward(Tensor3D queryInput, Tensor3D keyInput,
                             Tensor3D valueInput, float[][] mask) {

        int batchSize = queryInput.batch;
        int sqLen     = queryInput.rows;  // query sequence length
        int skLen     = keyInput.rows;    // key/value sequence length

        if (debug) {
            System.out.printf("[MultiHeadAttention] Q_in: [%d,%d,%d], K_in: [%d,%d,%d], " +
                              "V_in: [%d,%d,%d]%n",
                queryInput.batch, queryInput.rows, queryInput.cols,
                keyInput.batch,   keyInput.rows,   keyInput.cols,
                valueInput.batch, valueInput.rows, valueInput.cols);
        }

        // ── Step 1: Linear projections ────────────────────────────────────
        // Q = queryInput · W_Q^T   [B × S_q × dModel]
        // K = keyInput   · W_K^T   [B × S_k × dModel]
        // V = valueInput · W_V^T   [B × S_k × dModel]
        Tensor3D Q = wQ.forward(queryInput);  // [B × S_q × dModel]
        Tensor3D K = wK.forward(keyInput);    // [B × S_k × dModel]
        Tensor3D V = wV.forward(valueInput);  // [B × S_k × dModel]

        if (debug) {
            System.out.printf("[MultiHeadAttention] After projection - Q: [%d,%d,%d], " +
                              "K: [%d,%d,%d], V: [%d,%d,%d]%n",
                Q.batch, Q.rows, Q.cols, K.batch, K.rows, K.cols, V.batch, V.rows, V.cols);
        }

        // ── Step 2: Split into h heads ────────────────────────────────────
        // Each tensor is split along the last (cols) dimension into h slices.
        // Result: h tensors each [B × S × dHead]
        Tensor3D[] Qheads = Q.splitLastDim(numHeads);  // h × [B × S_q × dHead]
        Tensor3D[] Kheads = K.splitLastDim(numHeads);  // h × [B × S_k × dHead]
        Tensor3D[] Vheads = V.splitLastDim(numHeads);  // h × [B × S_k × dHead]

        if (debug) {
            System.out.printf("[MultiHeadAttention] Split into %d heads, each [%d,%d,%d]%n",
                numHeads, batchSize, sqLen, dHead);
        }

        // ── Step 3: Apply attention to each head ──────────────────────────
        Tensor3D[] headOutputs = new Tensor3D[numHeads];

        for (int h = 0; h < numHeads; h++) {
            headOutputs[h] = new Tensor3D(batchSize, sqLen, dHead);

            for (int b = 0; b < batchSize; b++) {
                // Extract 2-D matrices for batch item b, head h.
                Matrix Qb = Qheads[h].getMatrix(b);  // [S_q × dHead]
                Matrix Kb = Kheads[h].getMatrix(b);  // [S_k × dHead]
                Matrix Vb = Vheads[h].getMatrix(b);  // [S_k × dHead]

                // Apply scaled dot-product attention.
                // Attention(Q_h, K_h, V_h) → [S_q × dHead]
                Matrix attended = ScaledDotProductAttention.compute(Qb, Kb, Vb, mask);
                headOutputs[h].setMatrix(b, attended);
            }
        }

        // ── Step 4: Concatenate heads ─────────────────────────────────────
        // Each head produced [B × S_q × dHead].
        // Concat along last dim → [B × S_q × (h × dHead)] = [B × S_q × dModel]
        Tensor3D concatenated = Tensor3D.concatLastDim(headOutputs);  // [B × S_q × dModel]

        if (debug) {
            System.out.printf("[MultiHeadAttention] After concat: [%d,%d,%d]%n",
                concatenated.batch, concatenated.rows, concatenated.cols);
        }

        // ── Step 5: Output projection ─────────────────────────────────────
        // output = concatenated · W_O^T   [B × S_q × dModel]
        Tensor3D output = wO.forward(concatenated);

        if (debug) {
            System.out.printf("[MultiHeadAttention] Output: [%d,%d,%d]%n",
                output.batch, output.rows, output.cols);
        }

        return output;
    }
}
