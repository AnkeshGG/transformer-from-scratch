/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.embedding;

import transformer.TransformerConfig;
import transformer.tensor.Tensor3D;
import transformer.utils.RandomUtils;

import java.util.Random;

/**
 * TokenEmbedding — Trainable embedding lookup table for token IDs.
 *
 * <h2>What this represents</h2>
 * <p>The embedding layer maps discrete integer token IDs to continuous vector
 * representations. Each token in the vocabulary has a corresponding learnable
 * row vector of dimension {@code dModel}.
 *
 * <h2>Mathematical operation</h2>
 * <p>Given an input token ID sequence {@code [t_0, t_1, ..., t_{S-1}]}, the
 * embedding lookup simply retrieves rows from the embedding matrix E:
 * <pre>
 *   E ∈ R^{vocabSize × dModel}
 *   embed(t_i) = E[t_i]  ∈ R^{dModel}
 * </pre>
 *
 * <p>The full output for a batch of sequences is:
 * <pre>
 *   Input:  [B × S]         (batch of integer token ID sequences)
 *   Output: [B × S × dModel] (batch of embedding sequences)
 * </pre>
 *
 * <h2>Scaling</h2>
 * <p>Following the "Attention Is All You Need" paper (section 3.4), embeddings
 * are multiplied by {@code sqrt(dModel)} before adding positional encodings.
 * This scaling is applied here to keep the two contributions on a comparable scale.
 *
 * <h2>Initialisation</h2>
 * <p>The embedding matrix is initialised with small normal random values
 * (mean=0, std=0.02 is common for language models; we use 1/sqrt(dModel)
 * so that the initial embeddings have unit variance, consistent with the
 * expected scale after the sqrt(dModel) multiplication).
 */
public final class TokenEmbedding {

    /** Embedding weight matrix: shape [vocabSize × dModel]. */
    private final float[][] embeddingMatrix;

    /** Model dimension — the size of each embedding vector. */
    private final int dModel;

    /** Vocabulary size — total number of token types. */
    private final int vocabSize;

    /**
     * Scale factor applied to embeddings before adding positional encoding.
     * Paper: multiply by sqrt(dModel).
     */
    private final float scale;

    /** Whether debug mode is on (prints shapes). */
    private final boolean debug;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Construct a TokenEmbedding from a TransformerConfig.
     *
     * @param config model configuration
     */
    public TokenEmbedding(TransformerConfig config) {
        this.dModel    = config.dModel;
        this.vocabSize = config.vocabSize;
        this.scale     = (float) Math.sqrt(dModel);
        this.debug     = config.debugMode;

        // Initialise embedding matrix with normal random values.
        // std = 1/sqrt(dModel) → after ×scale, embeddings have ~unit variance.
        this.embeddingMatrix = new float[vocabSize][dModel];
        Random rng = RandomUtils.createRng(config.randomSeed);
        RandomUtils.normalInit(embeddingMatrix, 1.0 / Math.sqrt(dModel), rng);
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    /**
     * Embed a batch of token ID sequences.
     *
     * <p>Operation: for each token ID in each sequence, look up the corresponding
     * row in the embedding matrix, then scale by sqrt(dModel).
     *
     * @param tokenIds integer token IDs, shape [batchSize × sequenceLength]
     * @return embedded sequences, shape [batchSize × sequenceLength × dModel]
     * @throws IllegalArgumentException if any token ID is out of vocabulary range
     */
    public Tensor3D forward(int[][] tokenIds) {
        int batchSize = tokenIds.length;
        int seqLen    = tokenIds[0].length;

        if (debug) {
            System.out.printf("[TokenEmbedding] input shape: [%d, %d]%n", batchSize, seqLen);
        }

        Tensor3D output = new Tensor3D(batchSize, seqLen, dModel);

        for (int b = 0; b < batchSize; b++) {
            if (tokenIds[b].length != seqLen)
                throw new IllegalArgumentException(
                    "Batch item " + b + " has sequence length " + tokenIds[b].length +
                    " but expected " + seqLen);

            for (int s = 0; s < seqLen; s++) {
                int tokenId = tokenIds[b][s];

                if (tokenId < 0 || tokenId >= vocabSize)
                    throw new IllegalArgumentException(
                        "Token ID " + tokenId + " at [batch=" + b + ", pos=" + s + "] is out of " +
                        "vocabulary range [0, " + vocabSize + ")");

                // Look up embedding row and scale by sqrt(dModel).
                float[] embedding = embeddingMatrix[tokenId];
                for (int d = 0; d < dModel; d++) {
                    output.data[b][s][d] = embedding[d] * scale;
                }
            }
        }

        if (debug) {
            System.out.printf("[TokenEmbedding] output shape: [%d, %d, %d]%n",
                batchSize, seqLen, dModel);
        }

        return output;
    }

    // ── Direct weight access (for weight tying) ──────────────────────────────

    /**
     * Return the embedding matrix [vocabSize × dModel].
     * Can be used for weight tying with the output projection layer
     * (sharing weights between input embedding and output linear projection
     * is a common practice from the paper, section 3.4).
     */
    public float[][] getEmbeddingMatrix() {
        return embeddingMatrix;
    }

    /**
     * Get the embedding vector for a single token ID (without scale factor).
     * Useful for inspection/debugging.
     */
    public float[] getEmbedding(int tokenId) {
        if (tokenId < 0 || tokenId >= vocabSize)
            throw new IllegalArgumentException(
                "Token ID " + tokenId + " out of vocabulary range [0, " + vocabSize + ")");
        return embeddingMatrix[tokenId];
    }
}
