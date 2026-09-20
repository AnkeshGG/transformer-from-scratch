/*
 * transformer-from-scratch
 * MIT License — see LICENSE file at project root
 */

/**
 * Tensor primitives used throughout the Transformer.
 *
 * <ul>
 *   <li>{@link transformer.tensor.Matrix} — 2-D {@code float[][]} array with
 *       matrix multiply, transpose, softmax, and element-wise operations.
 *   <li>{@link transformer.tensor.Tensor3D} — 3-D {@code float[][][]} array
 *       ({@code [batch × rows × cols]}) with batch matrix multiply, head
 *       splitting/concatenation, and reshape.
 * </ul>
 *
 * <p>All operations return new objects; inputs are never mutated.
 */
package transformer.tensor;
