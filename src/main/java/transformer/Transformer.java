/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer;

import transformer.decoder.Decoder;
import transformer.encoder.Encoder;
import transformer.layers.Linear;
import transformer.tensor.Tensor3D;
import transformer.utils.MathUtils;
import transformer.utils.RandomUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Transformer — The complete encoder-decoder Transformer model.
 *
 * <h2>Architecture overview</h2>
 * <p>This is an implementation of the original architecture from:
 * "Attention Is All You Need" — Vaswani et al., 2017.
 *
 * <pre>
 *   Source tokens [B × S]
 *         ↓
 *   ┌─────────────────────┐
 *   │       ENCODER       │
 *   │  TokenEmbedding     │
 *   │  + PositionalEnc    │
 *   │  × N EncoderLayers  │
 *   └──────────┬──────────┘
 *              │ encoderOutput [B × S × dModel]
 *              │
 *   Target tokens [B × T]
 *         ↓
 *   ┌─────────────────────┐
 *   │       DECODER       │
 *   │  TokenEmbedding     │
 *   │  + PositionalEnc    │
 *   │  × N DecoderLayers  │
 *   │    (uses encoder    │
 *   │     output via      │
 *   │   cross-attention)  │
 *   └──────────┬──────────┘
 *              │ decoderOutput [B × T × dModel]
 *              ↓
 *   Linear projection  dModel → vocabSize
 *              ↓
 *   logits [B × T × vocabSize]
 *              ↓
 *   Softmax (row-wise over vocab dim)
 *              ↓
 *   probabilities [B × T × vocabSize]
 * </pre>
 *
 * <h2>Current implementation scope</h2>
 * <p>The full forward pass is implemented and produces correct probability
 * distributions. Backpropagation is not yet implemented; weight updates require
 * gradients through: softmax+CE, linear projection, layer norm, multi-head
 * attention, FFN, and embeddings. An Adam or SGD update step would follow.
 *
 * <h2>Greedy autoregressive generation</h2>
 * <p>{@link #generate} runs the standard inference loop:
 * encode source → start with BOS → argmax next token → append → repeat
 * until EOS or {@code maxLength} is reached.
 */
public final class Transformer {

    /** The encoder stack. */
    private final Encoder encoder;

    /** The decoder stack. */
    private final Decoder decoder;

    /**
     * Output projection: maps decoder hidden states to vocabulary logits.
     * W_vocab ∈ R^{vocabSize × dModel}
     * logits = decoderOutput · W_vocab^T + bias
     */
    private final Linear outputProjection;

    /** Model configuration. */
    private final TransformerConfig config;

    /** Debug mode flag. */
    private final boolean debug;

    // ── Special token IDs ────────────────────────────────────────────────────

    /** Padding token ID. */
    private final int padTokenId;

    /** Beginning-of-sequence token ID. */
    private final int bosTokenId;

    /** End-of-sequence token ID. */
    private final int eosTokenId;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct the full Transformer model.
     *
     * @param config     model configuration
     * @param padTokenId padding token ID (used for masking)
     * @param bosTokenId beginning-of-sequence token ID
     * @param eosTokenId end-of-sequence token ID
     */
    public Transformer(TransformerConfig config,
                       int padTokenId, int bosTokenId, int eosTokenId) {
        this.config     = config;
        this.padTokenId = padTokenId;
        this.bosTokenId = bosTokenId;
        this.eosTokenId = eosTokenId;
        this.debug      = config.debugMode;

        // All layers share the same RNG seeded from config for reproducibility.
        Random rng = RandomUtils.createRng(config.randomSeed);

        this.encoder          = new Encoder(config, padTokenId, rng);
        this.decoder          = new Decoder(config, padTokenId, rng);

        // Output projection: dModel → vocabSize
        this.outputProjection = new Linear(config.dModel, config.vocabSize, rng);
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Full Transformer forward pass (used during training with teacher forcing).
     *
     * <p>The target sequence passed here is the "shifted-right" version:
     * during training, if the expected output is [w₁, w₂, w₃, EOS], the decoder
     * input is [BOS, w₁, w₂, w₃]. This is teacher forcing.
     *
     * @param sourceTokenIds  source (encoder) token IDs [B × S]
     * @param targetTokenIds  target (decoder) token IDs, shifted right [B × T]
     * @return output probabilities [B × T × vocabSize]
     */
    public Tensor3D forward(int[][] sourceTokenIds, int[][] targetTokenIds) {
        return forwardWithLogits(sourceTokenIds, targetTokenIds)[0];
    }

    /**
     * Forward pass that returns both probabilities and logits (needed for loss computation).
     *
     * @param sourceTokenIds source token IDs [B × S]
     * @param targetTokenIds target token IDs [B × T]
     * @return Tensor3D[0] = probabilities [B × T × vocabSize],
     *         Tensor3D[1] = logits [B × T × vocabSize]
     */
    public Tensor3D[] forwardWithLogits(int[][] sourceTokenIds, int[][] targetTokenIds) {

        if (debug) {
            System.out.println("═══════════════════════════════════════════");
            System.out.println("           TRANSFORMER FORWARD PASS        ");
            System.out.println("═══════════════════════════════════════════");
        }

        // ── Step 1: Encode source sequence ────────────────────────────
        // Encoder: source tokens → encoder hidden states
        // encoderOutput: [B × S × dModel]
        Tensor3D encoderOutput = encoder.forward(sourceTokenIds);

        if (debug) {
            System.out.printf("[Transformer] Encoder output: [%d,%d,%d]%n",
                encoderOutput.batch, encoderOutput.rows, encoderOutput.cols);
        }

        // ── Step 2: Decode target sequence ────────────────────────────
        // Decoder: target tokens + encoder output → decoder hidden states
        // decoderOutput: [B × T × dModel]
        Tensor3D decoderOutput = decoder.forward(targetTokenIds, encoderOutput, sourceTokenIds);

        if (debug) {
            System.out.printf("[Transformer] Decoder output: [%d,%d,%d]%n",
                decoderOutput.batch, decoderOutput.rows, decoderOutput.cols);
        }

        // ── Step 3: Output linear projection ─────────────────────────
        // Project dModel → vocabSize for each position.
        // logits: [B × T × vocabSize]
        Tensor3D logits = outputProjection.forward(decoderOutput);

        if (debug) {
            System.out.printf("[Transformer] Logits shape: [%d,%d,%d]%n",
                logits.batch, logits.rows, logits.cols);
        }

        // ── Step 4: Numerically stable softmax ───────────────────────
        // Apply softmax over the vocabulary dimension (last dim) for each
        // batch item and position.
        // probabilities: [B × T × vocabSize]
        Tensor3D probabilities = applySoftmaxOverVocab(logits);

        if (debug) {
            System.out.printf("[Transformer] Probabilities shape: [%d,%d,%d]%n",
                probabilities.batch, probabilities.rows, probabilities.cols);
        }

        return new Tensor3D[]{probabilities, logits};
    }

    // ── Greedy generation ─────────────────────────────────────────────────────

    /**
     * Generate a target sequence from a source sequence using greedy decoding.
     *
     * <h2>Algorithm</h2>
     * <ol>
     *   <li>Encode the source sequence once.
     *   <li>Start with decoder input = [BOS].
     *   <li>Run the decoder, get probabilities for the <em>last</em> position.
     *   <li>Take the argmax → predicted next token.
     *   <li>Append to decoder input, repeat until EOS or maxLength.
     * </ol>
     *
     * <h2>Greedy vs beam search</h2>
     * <p>Greedy decoding always selects the single most probable token at each step.
     * This is fast but suboptimal (a globally better sequence may require locally
     * lower-probability choices). Beam search explores multiple candidates but is
     * more complex — it can be added as a future extension.
     *
     * @param sourceTokenIds source token IDs [1 × S] (single sequence, batch=1)
     * @param maxLength      maximum number of tokens to generate
     * @return generated token IDs (not including BOS, may include EOS)
     */
    public int[] generate(int[] sourceTokenIds, int maxLength) {

        if (debug) {
            System.out.println("═══════════════════════════════════════════");
            System.out.println("          AUTOREGRESSIVE GENERATION        ");
            System.out.println("═══════════════════════════════════════════");
        }

        // Wrap as batch of 1.
        int[][] sourceBatch = new int[][]{sourceTokenIds};

        // ── Step 1: Encode source once ────────────────────────────────
        Tensor3D encoderOutput = encoder.forward(sourceBatch);  // [1 × S × dModel]

        if (debug) {
            System.out.printf("[Generate] Encoder output: [%d,%d,%d]%n",
                encoderOutput.batch, encoderOutput.rows, encoderOutput.cols);
        }

        // ── Step 2: Initialise decoder input with BOS ─────────────────
        List<Integer> generated = new ArrayList<>();
        generated.add(bosTokenId);

        // ── Step 3: Autoregressive loop ───────────────────────────────
        for (int step = 0; step < maxLength; step++) {

            // Build current decoder input from generated tokens.
            int[] decoderInput = new int[generated.size()];
            for (int i = 0; i < generated.size(); i++)
                decoderInput[i] = generated.get(i);
            int[][] decoderBatch = new int[][]{decoderInput};

            // Run decoder.
            Tensor3D decoderOutput = decoder.forward(
                decoderBatch, encoderOutput, sourceBatch);  // [1 × curLen × dModel]

            // Project and softmax for the LAST position only.
            // The last position is the prediction for the next token.
            int curLen     = decoderOutput.rows;
            float[] lastHidden = new float[config.dModel];
            for (int d = 0; d < config.dModel; d++)
                lastHidden[d] = decoderOutput.get(0, curLen - 1, d);

            // Linear projection of last hidden state to vocabulary logits.
            float[] logits = projectToVocab(lastHidden);

            // Numerically stable softmax.
            float[] probs = MathUtils.softmax(logits);

            // Greedy: pick token with highest probability.
            int nextToken = MathUtils.argmax(probs);

            if (debug) {
                System.out.printf("[Generate] Step %d: predicted token=%d, prob=%.4f%n",
                    step + 1, nextToken, probs[nextToken]);
            }

            // If EOS generated, stop.
            if (nextToken == eosTokenId) {
                generated.add(eosTokenId);
                break;
            }

            generated.add(nextToken);
        }

        // Remove BOS from the output (keep only generated tokens).
        int[] result = new int[generated.size() - 1];
        for (int i = 1; i < generated.size(); i++)
            result[i - 1] = generated.get(i);

        return result;
    }

    // ── Helper methods ────────────────────────────────────────────────────────

    /**
     * Apply softmax over the last (vocabulary) dimension of a 3-D tensor.
     *
     * <p>For each batch item b and sequence position t, applies softmax over
     * the vocabSize dimension:
     * <pre>
     *   probs[b][t] = softmax(logits[b][t])  where logits[b][t] ∈ R^{vocabSize}
     * </pre>
     *
     * @param logits [B × T × vocabSize]
     * @return probabilities [B × T × vocabSize], each row sums to 1.0
     */
    private Tensor3D applySoftmaxOverVocab(Tensor3D logits) {
        Tensor3D result = new Tensor3D(logits.batch, logits.rows, logits.cols);
        for (int b = 0; b < logits.batch; b++) {
            for (int t = 0; t < logits.rows; t++) {
                // Extract logits for this position.
                float[] posLogits = logits.data[b][t];
                // Apply stable softmax.
                float[] posProbs = MathUtils.softmax(posLogits);
                // Write back.
                result.data[b][t] = posProbs;
            }
        }
        return result;
    }

    /**
     * Project a single hidden state vector [dModel] to vocabulary logits [vocabSize].
     * Used during autoregressive generation for the last position.
     */
    private float[] projectToVocab(float[] hidden) {
        // Manual matmul: logits[v] = Σ_d hidden[d] * weight[v][d] + bias[v]
        float[] logits = new float[config.vocabSize];
        for (int v = 0; v < config.vocabSize; v++) {
            double sum = outputProjection.bias[v];
            for (int d = 0; d < config.dModel; d++) {
                sum += hidden[d] * outputProjection.weight[v][d];
            }
            logits[v] = (float) sum;
        }
        return logits;
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    /** Return the encoder for direct access. */
    public Encoder getEncoder() { return encoder; }

    /** Return the decoder for direct access. */
    public Decoder getDecoder() { return decoder; }

    /** Return the model configuration. */
    public TransformerConfig getConfig() { return config; }
}
