/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */

/**
 * Tokenization utilities.
 *
 * <p>{@link transformer.tokenizer.SimpleTokenizer} is a word-level tokenizer
 * suitable for demos and small experiments. For production use, consider
 * replacing with a BPE or SentencePiece implementation.
 *
 * <p>Special token IDs are fixed:
 * <pre>
 *   0 = &lt;PAD&gt;   1 = &lt;BOS&gt;   2 = &lt;EOS&gt;   3 = &lt;UNK&gt;
 * </pre>
 */
package transformer.tokenizer;
