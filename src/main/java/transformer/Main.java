/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer;

import transformer.attention.AttentionMask;
import transformer.loss.CrossEntropyLoss;
import transformer.tensor.Tensor3D;
import transformer.tokenizer.SimpleTokenizer;
import transformer.utils.MathUtils;

import java.util.Arrays;
import java.util.List;

/**
 * Main — End-to-end demonstration of the Transformer implementation.
 *
 * <h2>What this demo shows</h2>
 * <ol>
 *   <li>Building a tiny vocabulary and tokenizer
 *   <li>Constructing a tiny Transformer (dModel=8, 2 heads, 2 layers)
 *   <li>Encoding a source sentence "I love cats"
 *   <li>Running the full forward pass with teacher forcing
 *   <li>Printing all intermediate tensor shapes
 *   <li>Showing the output probability distribution
 *   <li>Computing cross-entropy loss
 *   <li>Running autoregressive generation and printing the predicted tokens
 *   <li>Showing the causal attention mask
 * </ol>
 *
 * <h2>Expected output (shapes are the key thing to verify)</h2>
 * <pre>
 *   [Encoder] output shape: [1, 3, 8]
 *   [Decoder] output shape: [1, 4, 8]
 *   [Transformer] Logits shape: [1, 4, 7]
 *   [Transformer] Probabilities shape: [1, 4, 7]
 * </pre>
 *
 * <p>Run with: {@code mvn compile exec:java -Dexec.mainClass=transformer.Main}
 */
public class Main {

    public static void main(String[] args) {

        System.out.println("╔══════════════════════════════════════════════════╗");
        System.out.println("║   Transformer from Scratch — Java Implementation ║");
        System.out.println("║   Based on: Attention Is All You Need (2017)     ║");
        System.out.println("╚══════════════════════════════════════════════════╝");
        System.out.println();

        // ══════════════════════════════════════════════════════════════════
        // 1. VOCABULARY AND TOKENIZER
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 1: Build Vocabulary ─────────────────────");
        // Special tokens are added automatically: PAD=0, BOS=1, EOS=2, UNK=3
        List<String> vocab = List.of("I", "love", "cats", "dogs");
        SimpleTokenizer tokenizer = new SimpleTokenizer(vocab);
        tokenizer.printVocabulary();
        System.out.println();

        // ══════════════════════════════════════════════════════════════════
        // 2. TRANSFORMER CONFIGURATION
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 2: Build Transformer Configuration ──────");
        // Tiny Transformer for demonstration.
        // vocabSize=8  (4 special + 4 words)
        // dModel=8, numHeads=2, dFF=16, numLayers=2
        int vocabSize = tokenizer.vocabSize();  // 8
        TransformerConfig config = TransformerConfig.tiny(vocabSize);
        System.out.println(config);
        System.out.println("  dHead = dModel / numHeads = " + config.dHead);
        System.out.println();

        // ══════════════════════════════════════════════════════════════════
        // 3. BUILD THE TRANSFORMER
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 3: Build Transformer Model ──────────────");
        Transformer transformer = new Transformer(
            config,
            SimpleTokenizer.PAD_ID,   // 0
            SimpleTokenizer.BOS_ID,   // 1
            SimpleTokenizer.EOS_ID    // 2
        );
        System.out.println("Transformer built successfully.");
        System.out.printf("  Encoder layers: %d%n", config.numEncoderLayers);
        System.out.printf("  Decoder layers: %d%n", config.numDecoderLayers);
        System.out.println();

        // ══════════════════════════════════════════════════════════════════
        // 4. TOKENIZE SOURCE AND TARGET
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 4: Tokenize Input/Output ───────────────");

        // Source: "I love cats"  (encoder input)
        // Token IDs without special tokens (encoder doesn't need BOS/EOS in our demo)
        String sourceText = "I love cats";
        int[] sourceIds = tokenizer.encodeRaw(sourceText);
        System.out.println("Source text:   \"" + sourceText + "\"");
        System.out.println("Source IDs:    " + Arrays.toString(sourceIds));

        // Target (decoder input): <BOS> I love cats  (teacher forcing input)
        // Target (label):          I love cats <EOS> (what we want to predict)
        String targetText = "I love cats";
        List<Integer> targetEncoded = tokenizer.encode(targetText);  // [BOS, I, love, cats, EOS]
        int[] targetIds = SimpleTokenizer.toArray(targetEncoded);
        System.out.println("Target text:   \"" + targetText + "\"");
        System.out.println("Target IDs (with BOS/EOS): " + Arrays.toString(targetIds));

        // Decoder input = everything except last token: [BOS, I, love, cats]
        int[] decoderInputIds = Arrays.copyOfRange(targetIds, 0, targetIds.length - 1);
        // Labels = everything except first token: [I, love, cats, EOS]
        int[] labelIds = Arrays.copyOfRange(targetIds, 1, targetIds.length);

        System.out.println("Decoder input: " + Arrays.toString(decoderInputIds)
            + " = " + Arrays.toString(tokensToStrings(decoderInputIds, tokenizer)));
        System.out.println("Labels:        " + Arrays.toString(labelIds)
            + " = " + Arrays.toString(tokensToStrings(labelIds, tokenizer)));
        System.out.println();

        // Wrap as batch of 1.
        int[][] sourceBatch  = new int[][]{sourceIds};
        int[][] decoderBatch = new int[][]{decoderInputIds};

        // ══════════════════════════════════════════════════════════════════
        // 5. SHOW CAUSAL MASK
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 5: Causal Attention Mask ───────────────");
        int tgtLen = decoderInputIds.length;
        float[][] causalMask = AttentionMask.causalMask(tgtLen);
        System.out.println("Causal mask for decoder self-attention (T=" + tgtLen + "):");
        System.out.print(AttentionMask.formatMask(causalMask));
        System.out.println("Position i can only attend to positions j ≤ i.");
        System.out.println();

        // ══════════════════════════════════════════════════════════════════
        // 6. FULL FORWARD PASS
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 6: Full Transformer Forward Pass ────────");
        System.out.println("  Source tokens: " + Arrays.toString(sourceIds)
            + " → Encoder → [1, " + sourceIds.length + ", " + config.dModel + "]");
        System.out.println("  Decoder input: " + Arrays.toString(decoderInputIds)
            + " → Decoder → [1, " + decoderInputIds.length + ", " + config.dModel + "]");
        System.out.println();

        Tensor3D[] results = transformer.forwardWithLogits(sourceBatch, decoderBatch);
        Tensor3D probabilities = results[0];  // [1 × T × vocabSize]
        Tensor3D logits        = results[1];  // [1 × T × vocabSize]

        System.out.println();
        System.out.printf("✓ Logits shape:        [%d, %d, %d]%n",
            logits.batch, logits.rows, logits.cols);
        System.out.printf("✓ Probabilities shape: [%d, %d, %d]%n",
            probabilities.batch, probabilities.rows, probabilities.cols);
        System.out.println();

        // ══════════════════════════════════════════════════════════════════
        // 7. PROBABILITY DISTRIBUTION AT EACH POSITION
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 7: Probability Distribution ────────────");
        System.out.println("(Predictions are RANDOM at this stage — no training has occurred)");
        System.out.println();

        for (int t = 0; t < probabilities.rows; t++) {
            float[] probs = probabilities.data[0][t];
            int predictedId = MathUtils.argmax(probs);
            System.out.printf("  Position %d (input='%s') → predicted next='%s' (ID=%d, prob=%.4f)%n",
                t,
                tokenizer.idToWord(decoderInputIds[t]),
                tokenizer.idToWord(predictedId),
                predictedId,
                probs[predictedId]);

            // Print full distribution.
            System.out.print("    Distribution: ");
            for (int v = 0; v < vocabSize; v++) {
                System.out.printf("%s:%.3f ", tokenizer.idToWord(v), probs[v]);
            }
            System.out.println();

            // Verify softmax sums to 1.
            float sum = 0;
            for (float p : probs) sum += p;
            System.out.printf("    Sum of probs: %.6f (should be ≈ 1.0)%n", sum);
        }
        System.out.println();

        // ══════════════════════════════════════════════════════════════════
        // 8. CROSS-ENTROPY LOSS
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 8: Cross-Entropy Loss ───────────────────");
        CrossEntropyLoss lossFunction = new CrossEntropyLoss(SimpleTokenizer.PAD_ID);

        // Convert Tensor3D logits to float[][][]
        float[][][] logitsArray = new float[1][logits.rows][logits.cols];
        logitsArray[0] = logits.data[0];

        int[][] labelsBatch = new int[][]{labelIds};

        float loss = lossFunction.compute(logitsArray, labelsBatch);
        double perplexity = CrossEntropyLoss.perplexity(loss);

        System.out.printf("  Loss:       %.4f nats%n", loss);
        System.out.printf("  Perplexity: %.2f  (random baseline ≈ %.1f for vocab size %d)%n",
            perplexity, (double) vocabSize, vocabSize);
        System.out.println("  (High perplexity is expected before training)");
        System.out.println();

        // ══════════════════════════════════════════════════════════════════
        // 9. AUTOREGRESSIVE GENERATION
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 9: Autoregressive Generation ────────────");
        System.out.println("  Source: \"" + sourceText + "\"");
        System.out.println("  Generating up to 10 tokens with greedy decoding...");

        // Disable debug for generation to reduce output.
        TransformerConfig genConfig = new TransformerConfig.Builder()
            .dModel(config.dModel)
            .numHeads(config.numHeads)
            .dFF(config.dFF)
            .numEncoderLayers(config.numEncoderLayers)
            .numDecoderLayers(config.numDecoderLayers)
            .vocabSize(config.vocabSize)
            .maxSequenceLength(config.maxSequenceLength)
            .randomSeed(config.randomSeed)
            .debugMode(false)
            .build();

        Transformer genTransformer = new Transformer(
            genConfig,
            SimpleTokenizer.PAD_ID,
            SimpleTokenizer.BOS_ID,
            SimpleTokenizer.EOS_ID
        );

        int[] generated = genTransformer.generate(sourceIds, 10);

        System.out.println("  Generated token IDs: " + Arrays.toString(generated));
        System.out.println("  Generated text:      \"" + tokenizer.decode(generated) + "\"");
        System.out.println("  (Random output is expected — model has not been trained)");
        System.out.println();

        // ══════════════════════════════════════════════════════════════════
        // 10. SANITY CHECKS SUMMARY
        // ══════════════════════════════════════════════════════════════════

        System.out.println("─── Step 10: Sanity Checks ───────────────────────");

        // Check encoder output shape
        boolean ok = true;

        // Re-run encoder alone for shape check.
        Tensor3D encOut = transformer.getEncoder().forward(sourceBatch);
        ok &= check("Encoder output shape [1," + sourceIds.length + "," + config.dModel + "]",
            encOut.batch == 1 && encOut.rows == sourceIds.length && encOut.cols == config.dModel);

        ok &= check("Logits shape [1," + tgtLen + "," + vocabSize + "]",
            logits.batch == 1 && logits.rows == tgtLen && logits.cols == vocabSize);

        ok &= check("Probabilities shape [1," + tgtLen + "," + vocabSize + "]",
            probabilities.batch == 1 && probabilities.rows == tgtLen && probabilities.cols == vocabSize);

        // Check each position's probabilities sum to 1.
        for (int t = 0; t < probabilities.rows; t++) {
            float sum = 0;
            for (float p : probabilities.data[0][t]) sum += p;
            ok &= check("Probs at position " + t + " sum to 1.0", Math.abs(sum - 1.0f) < 1e-4f);
        }

        // Check causal mask property.
        ok &= check("Causal mask: upper triangle is -inf",
            causalMask[0][1] == AttentionMask.MASK_VALUE &&
            causalMask[0][2] == AttentionMask.MASK_VALUE);
        ok &= check("Causal mask: diagonal is 0",
            causalMask[0][0] == 0.0f && causalMask[1][1] == 0.0f);

        System.out.println();
        if (ok) {
            System.out.println("✅ All sanity checks PASSED.");
        } else {
            System.out.println("❌ Some sanity checks FAILED. See above for details.");
        }

        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════╗");
        System.out.println("║  Demo complete! The forward pass is working.     ║");
        System.out.println("║  Next: add backpropagation to enable training.   ║");
        System.out.println("╚══════════════════════════════════════════════════╝");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static boolean check(String label, boolean condition) {
        System.out.printf("  [%s] %s%n", condition ? "✓" : "✗", label);
        return condition;
    }

    private static String[] tokensToStrings(int[] ids, SimpleTokenizer tokenizer) {
        String[] result = new String[ids.length];
        for (int i = 0; i < ids.length; i++) result[i] = tokenizer.idToWord(ids[i]);
        return result;
    }
}
