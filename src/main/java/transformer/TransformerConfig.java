/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer;

/**
 * TransformerConfig — Immutable configuration record for the full Transformer model.
 *
 * <p>All hyper-parameters are gathered here so that every layer can be constructed
 * from a single config object, making experimentation easy (just change one place).
 *
 * <p>Mathematical relationship between the fields:
 * <pre>
 *   dHead   = dModel / numHeads   (per-head key/query/value dimension)
 *   dModel  = numHeads × dHead    (embedding dimension throughout the model)
 *   dFF     ≈ 4 × dModel          (typical feed-forward inner dimension)
 * </pre>
 *
 * <p>Example (paper defaults):
 * <pre>
 *   dModel   = 512
 *   numHeads = 8      → dHead = 64
 *   dFF      = 2048
 *   numLayers = 6
 * </pre>
 *
 * <p>Example (tiny, for unit tests):
 * <pre>
 *   dModel   = 8
 *   numHeads = 2      → dHead = 4
 *   dFF      = 16
 *   numLayers = 2
 * </pre>
 */
public final class TransformerConfig {

    // ── Model dimensions ────────────────────────────────────────────────────

    /** Embedding / hidden dimension throughout the model. Must be divisible by numHeads. */
    public final int dModel;

    /** Number of attention heads in each Multi-Head Attention layer. */
    public final int numHeads;

    /**
     * Per-head dimension.
     * Derived: dHead = dModel / numHeads.
     * Explicitly stored to avoid recomputation and for clarity.
     */
    public final int dHead;

    /** Inner dimension of the position-wise feed-forward network (typically 4 × dModel). */
    public final int dFF;

    // ── Stacking ────────────────────────────────────────────────────────────

    /** Number of encoder layers stacked (N in the paper). */
    public final int numEncoderLayers;

    /** Number of decoder layers stacked (N in the paper). */
    public final int numDecoderLayers;

    // ── Vocabulary & sequence ───────────────────────────────────────────────

    /** Total vocabulary size (number of unique tokens including special tokens). */
    public final int vocabSize;

    /** Maximum sequence length supported by positional encoding. */
    public final int maxSequenceLength;

    // ── Regularisation ──────────────────────────────────────────────────────

    /**
     * Dropout rate p ∈ [0, 1). Set to 0.0 to disable dropout (default for inference).
     * Stored for use when training support is added.
     */
    public final double dropoutRate;

    // ── Numerical stability ─────────────────────────────────────────────────

    /** Small value added to variances in LayerNorm to avoid division by zero. */
    public final double layerNormEpsilon;

    // ── Reproducibility ─────────────────────────────────────────────────────

    /** Random seed for weight initialisation. Use same seed for reproducible runs. */
    public final long randomSeed;

    // ── Debug ───────────────────────────────────────────────────────────────

    /** When true, layers print shape information during forward pass. */
    public final boolean debugMode;

    // ── Constructor ─────────────────────────────────────────────────────────

    private TransformerConfig(Builder builder) {
        this.dModel            = builder.dModel;
        this.numHeads          = builder.numHeads;
        this.dHead             = builder.dModel / builder.numHeads;
        this.dFF               = builder.dFF;
        this.numEncoderLayers  = builder.numEncoderLayers;
        this.numDecoderLayers  = builder.numDecoderLayers;
        this.vocabSize         = builder.vocabSize;
        this.maxSequenceLength = builder.maxSequenceLength;
        this.dropoutRate       = builder.dropoutRate;
        this.layerNormEpsilon  = builder.layerNormEpsilon;
        this.randomSeed        = builder.randomSeed;
        this.debugMode         = builder.debugMode;

        validate();
    }

    /** Validate consistency of configuration. */
    private void validate() {
        if (dModel <= 0)
            throw new IllegalArgumentException("dModel must be positive, got: " + dModel);
        if (numHeads <= 0)
            throw new IllegalArgumentException("numHeads must be positive, got: " + numHeads);
        if (dModel % numHeads != 0)
            throw new IllegalArgumentException(
                    "dModel (" + dModel + ") must be divisible by numHeads (" + numHeads + ")");
        if (dFF <= 0)
            throw new IllegalArgumentException("dFF must be positive, got: " + dFF);
        if (numEncoderLayers <= 0)
            throw new IllegalArgumentException("numEncoderLayers must be positive");
        if (numDecoderLayers <= 0)
            throw new IllegalArgumentException("numDecoderLayers must be positive");
        if (vocabSize <= 0)
            throw new IllegalArgumentException("vocabSize must be positive");
        if (maxSequenceLength <= 0)
            throw new IllegalArgumentException("maxSequenceLength must be positive");
        if (dropoutRate < 0 || dropoutRate >= 1)
            throw new IllegalArgumentException("dropoutRate must be in [0, 1), got: " + dropoutRate);
        if (layerNormEpsilon <= 0)
            throw new IllegalArgumentException("layerNormEpsilon must be positive");
    }

    @Override
    public String toString() {
        return String.format(
            "TransformerConfig{dModel=%d, numHeads=%d, dHead=%d, dFF=%d, " +
            "encoderLayers=%d, decoderLayers=%d, vocabSize=%d, maxSeqLen=%d, " +
            "dropout=%.2f, seed=%d}",
            dModel, numHeads, dHead, dFF,
            numEncoderLayers, numDecoderLayers, vocabSize, maxSequenceLength,
            dropoutRate, randomSeed);
    }

    // ── Builder ─────────────────────────────────────────────────────────────

    /** Fluent builder for TransformerConfig. */
    public static final class Builder {
        // Mandatory — no defaults; callers must set these.
        private int dModel;
        private int numHeads;
        private int dFF;
        private int numEncoderLayers;
        private int numDecoderLayers;
        private int vocabSize;

        // Optional — sensible defaults.
        private int    maxSequenceLength = 512;
        private double dropoutRate       = 0.1;
        private double layerNormEpsilon  = 1e-6;
        private long   randomSeed        = 42L;
        private boolean debugMode        = false;

        public Builder dModel(int dModel)                     { this.dModel = dModel;                         return this; }
        public Builder numHeads(int numHeads)                 { this.numHeads = numHeads;                     return this; }
        public Builder dFF(int dFF)                           { this.dFF = dFF;                               return this; }
        public Builder numEncoderLayers(int n)                { this.numEncoderLayers = n;                    return this; }
        public Builder numDecoderLayers(int n)                { this.numDecoderLayers = n;                    return this; }
        public Builder vocabSize(int vocabSize)               { this.vocabSize = vocabSize;                   return this; }
        public Builder maxSequenceLength(int maxSeqLen)       { this.maxSequenceLength = maxSeqLen;           return this; }
        public Builder dropoutRate(double rate)               { this.dropoutRate = rate;                      return this; }
        public Builder layerNormEpsilon(double eps)           { this.layerNormEpsilon = eps;                  return this; }
        public Builder randomSeed(long seed)                  { this.randomSeed = seed;                       return this; }
        public Builder debugMode(boolean debug)               { this.debugMode = debug;                       return this; }

        public TransformerConfig build() {
            return new TransformerConfig(this);
        }
    }

    // ── Factory methods for common configurations ────────────────────────────

    /**
     * Returns the configuration from the original "Attention Is All You Need" paper (base model).
     * dModel=512, 8 heads, dFF=2048, 6 layers.
     */
    public static TransformerConfig paperBase(int vocabSize) {
        return new Builder()
                .dModel(512).numHeads(8).dFF(2048)
                .numEncoderLayers(6).numDecoderLayers(6)
                .vocabSize(vocabSize)
                .maxSequenceLength(512)
                .dropoutRate(0.1)
                .build();
    }

    /**
     * Returns a tiny configuration suitable for unit tests and demonstrations.
     * dModel=8, 2 heads, dFF=16, 2 layers.
     */
    public static TransformerConfig tiny(int vocabSize) {
        return new Builder()
                .dModel(8).numHeads(2).dFF(16)
                .numEncoderLayers(2).numDecoderLayers(2)
                .vocabSize(vocabSize)
                .maxSequenceLength(16)
                .dropoutRate(0.0)
                .debugMode(true)
                .build();
    }
}
