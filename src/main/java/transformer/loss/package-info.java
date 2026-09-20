/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */

/**
 * Loss functions.
 *
 * <p>{@link transformer.loss.CrossEntropyLoss} computes the mean cross-entropy
 * loss over a batch of sequence logits, skipping padding positions.
 * Uses numerically stable log-softmax internally.
 */
package transformer.loss;
