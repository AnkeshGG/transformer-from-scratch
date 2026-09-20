/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.loss;

import transformer.utils.MathUtils;

/**
 * CrossEntropyLoss — Numerically stable cross-entropy loss for sequence classification.
 *
 * <h2>Mathematical definition</h2>
 * <p>For a single prediction at one sequence position:
 * <pre>
 *   CE(logits, target) = −log(softmax(logits)[target])
 *                      = −log(exp(logits[target]) / Σ_v exp(logits[v]))
 *                      = −logits[target] + log(Σ_v exp(logits[v]))
 * </pre>
 *
 * <p>The stable computation uses log-softmax:
 * <pre>
 *   logSoftmax(x)[i] = x[i] − max(x) − log(Σ_j exp(x[j] − max(x)))
 *   CE(logits, target) = −logSoftmax(logits)[target]
 * </pre>
 *
 * <h2>Sequence cross-entropy (with padding)</h2>
 * <p>For a sequence of T positions with target labels t_0, ..., t_{T-1}:
 * <pre>
 *   SeqCE = (1 / numNonPad) × Σ_{t: t_t ≠ PAD} −log(softmax(logits_t)[t_t])
 * </pre>
 * Padding tokens (target = PAD_ID) do NOT contribute to the loss.
 * This ensures padding doesn't distort the learning signal.
 *
 * <h2>Usage in training</h2>
 * <p>With teacher forcing, the expected output at position t is the target token t_{t+1}.
 * The logits at position t are compared with the true next token.
 *
 * <h2>Gradient (for backpropagation)</h2>
 * <p>This class computes the scalar loss for monitoring and evaluation.
 * The gradient of CE with respect to logits has the well-known closed form:
 * <pre>
 *   dCE/dlogits[v] = softmax(logits)[v] − 1{v == target}
 * </pre>
 * (softmax output minus one-hot target vector).
 */
public final class CrossEntropyLoss {

    /** Token ID used for padding; positions with this target are excluded from the loss. */
    private final int padTokenId;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * @param padTokenId padding token ID (excluded from loss computation)
     */
    public CrossEntropyLoss(int padTokenId) {
        this.padTokenId = padTokenId;
    }

    // ── Loss computation ─────────────────────────────────────────────────────

    /**
     * Compute cross-entropy loss over a batch of sequence logits.
     *
     * @param logits  raw (unnormalised) logits [B × T × vocabSize]
     *                Each logits[b][t] is a vector of length vocabSize.
     * @param targets target token IDs [B × T]
     *                targets[b][t] is the correct token ID at position t of batch item b.
     *                Use padTokenId for positions that should not contribute to loss.
     * @return mean cross-entropy loss (scalar, averaged over non-padding positions)
     */
    public float compute(float[][][] logits, int[][] targets) {
        int batchSize = logits.length;
        int seqLen    = logits[0].length;

        if (targets.length != batchSize || targets[0].length != seqLen)
            throw new IllegalArgumentException(
                "CrossEntropyLoss: logits and targets batch/sequence dimensions do not match.");

        double totalLoss     = 0.0;
        int    numNonPadding = 0;

        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                int targetId = targets[b][t];

                // Skip padding positions.
                if (targetId == padTokenId) continue;

                if (targetId < 0 || targetId >= logits[b][t].length)
                    throw new IllegalArgumentException(
                        "CrossEntropyLoss: target ID " + targetId +
                        " at [" + b + "," + t + "] is out of vocab range [0," +
                        logits[b][t].length + ")");

                // Stable log-softmax.
                float[] logSoftmax = MathUtils.logSoftmax(logits[b][t]);

                // Cross-entropy: −log(prob(target)) = −logSoftmax(target)
                totalLoss += -logSoftmax[targetId];
                numNonPadding++;
            }
        }

        if (numNonPadding == 0)
            return 0.0f;

        return (float) (totalLoss / numNonPadding);
    }

    /**
     * Compute cross-entropy loss for a single sequence.
     * Convenience wrapper around {@link #compute(float[][][], int[][])}.
     *
     * @param logits  [T × vocabSize]  (single sequence)
     * @param targets [T]              (target token IDs for each position)
     * @return mean cross-entropy loss
     */
    public float computeSequence(float[][] logits, int[] targets) {
        return compute(new float[][][]{logits}, new int[][]{targets});
    }

    /**
     * Compute perplexity from cross-entropy loss.
     * <pre>
     *   Perplexity = exp(CrossEntropyLoss)
     * </pre>
     * Lower perplexity = better model. A random model over V tokens has perplexity V.
     *
     * @param loss cross-entropy loss (nats)
     * @return perplexity
     */
    public static double perplexity(float loss) {
        return Math.exp(loss);
    }
}
