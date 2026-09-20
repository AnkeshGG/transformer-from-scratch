/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */
package transformer.tokenizer;

import java.util.*;

/**
 * A minimal whitespace-splitting word-level tokenizer.
 *
 * <p>Suitable for demos and small experiments. For real applications, replace
 * with a BPE (GPT-style), WordPiece (BERT-style), or SentencePiece tokenizer.
 *
 * <h2>Special token IDs</h2>
 * <pre>
 *   0 = &lt;PAD&gt;  — padding (equalizes lengths in a batch)
 *   1 = &lt;BOS&gt;  — beginning of sequence
 *   2 = &lt;EOS&gt;  — end of sequence
 *   3 = &lt;UNK&gt;  — out-of-vocabulary word
 *   4+ = regular vocabulary words (insertion order)
 * </pre>
 *
 * <p>Encoding always inserts BOS at position 0 and EOS at the end.
 * The vocabulary is case-sensitive.
 */
public final class SimpleTokenizer {

    // ── Special token constants ──────────────────────────────────────────────

    public static final String PAD_TOKEN = "<PAD>";
    public static final String BOS_TOKEN = "<BOS>";
    public static final String EOS_TOKEN = "<EOS>";
    public static final String UNK_TOKEN = "<UNK>";

    public static final int PAD_ID = 0;
    public static final int BOS_ID = 1;
    public static final int EOS_ID = 2;
    public static final int UNK_ID = 3;

    // ── Vocabulary ───────────────────────────────────────────────────────────

    /** word → ID mapping. */
    private final Map<String, Integer> wordToId;

    /** ID → word mapping. */
    private final List<String> idToWord;

    // ── Constructor ─────────────────────────────────────────────────────────

    /**
     * Create a tokenizer with a predefined vocabulary.
     *
     * <p>The vocabulary list should NOT include special tokens — they are
     * added automatically at IDs 0–3.
     *
     * @param vocabulary list of vocabulary words (in order, starting at ID 4)
     */
    public SimpleTokenizer(List<String> vocabulary) {
        this.wordToId = new LinkedHashMap<>();
        this.idToWord = new ArrayList<>();

        // Add special tokens first (IDs 0–3).
        addToken(PAD_TOKEN);  // ID = 0
        addToken(BOS_TOKEN);  // ID = 1
        addToken(EOS_TOKEN);  // ID = 2
        addToken(UNK_TOKEN);  // ID = 3

        // Add vocabulary words.
        for (String word : vocabulary) {
            if (!wordToId.containsKey(word)) {
                addToken(word);
            }
        }
    }

    // ── Tokenization ─────────────────────────────────────────────────────────

    /**
     * Encode a text string into a list of token IDs.
     *
     * <p>Splits on whitespace, looks up each word, inserts BOS at start
     * and EOS at end.
     *
     * @param text input text
     * @param addSpecialTokens if true, prepend BOS and append EOS
     * @return list of token IDs
     */
    public List<Integer> encode(String text, boolean addSpecialTokens) {
        String[] words = text.trim().split("\\s+");
        List<Integer> ids = new ArrayList<>();

        if (addSpecialTokens) ids.add(BOS_ID);

        for (String word : words) {
            ids.add(wordToId.getOrDefault(word, UNK_ID));
        }

        if (addSpecialTokens) ids.add(EOS_ID);

        return ids;
    }

    /**
     * Encode text with special tokens added (convenience overload).
     */
    public List<Integer> encode(String text) {
        return encode(text, true);
    }

    /**
     * Convert a list of token IDs back to a string.
     * Special tokens are included in the output (caller can filter them).
     *
     * @param ids list of token IDs
     * @return decoded string
     */
    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ids.size(); i++) {
            int id = ids.get(i);
            if (id < 0 || id >= idToWord.size()) {
                sb.append(UNK_TOKEN);
            } else {
                sb.append(idToWord.get(id));
            }
            if (i < ids.size() - 1) sb.append(" ");
        }
        return sb.toString();
    }

    /**
     * Decode an int[] into a string (useful after generation).
     */
    public String decode(int[] ids) {
        List<Integer> list = new ArrayList<>();
        for (int id : ids) list.add(id);
        return decode(list);
    }

    /**
     * Convert a token ID to its string form.
     *
     * @param id token ID
     * @return word string, or {@link #UNK_TOKEN} if out of range
     */
    public String idToWord(int id) {
        if (id < 0 || id >= idToWord.size()) return UNK_TOKEN;
        return idToWord.get(id);
    }

    /**
     * Convert a word to its token ID.
     *
     * @param word input word
     * @return token ID, or {@link #UNK_ID} if not in vocabulary
     */
    public int wordToId(String word) {
        return wordToId.getOrDefault(word, UNK_ID);
    }

    /**
     * Encode text into a primitive int array (without special tokens).
     * Useful for creating decoder inputs without BOS/EOS.
     */
    public int[] encodeRaw(String text) {
        String[] words = text.trim().split("\\s+");
        int[] ids = new int[words.length];
        for (int i = 0; i < words.length; i++)
            ids[i] = wordToId.getOrDefault(words[i], UNK_ID);
        return ids;
    }

    /**
     * Convert a List&lt;Integer&gt; to int[].
     */
    public static int[] toArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }

    // ── Vocabulary info ───────────────────────────────────────────────────────

    /**
     * Total vocabulary size (including special tokens).
     */
    public int vocabSize() {
        return idToWord.size();
    }

    /**
     * Print the full vocabulary to stdout (for debugging).
     */
    public void printVocabulary() {
        System.out.println("Vocabulary (" + vocabSize() + " tokens):");
        for (int i = 0; i < idToWord.size(); i++) {
            System.out.printf("  %3d → %s%n", i, idToWord.get(i));
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private void addToken(String token) {
        int id = idToWord.size();
        wordToId.put(token, id);
        idToWord.add(token);
    }
}
