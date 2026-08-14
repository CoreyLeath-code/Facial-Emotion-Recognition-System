# Complexity analysis

Let \(B\) be batch size, \(H\times W\) input resolution, \(C_{in}, C_{out}\) channel counts, \(K\) convolution-kernel width, and \(d\) flattened feature dimension.

| Operation | Best | Average | Worst | Space |
| --- | ---: | ---: | ---: | ---: |
| Pixel-string parsing (FERDataset) | \(\Theta(HW)\) | \(\Theta(HW)\) | \(\Theta(HW)\) | \(\Theta(HW)\) per sample |
| 2-D convolution | \(\Theta(BHWC_{in}C_{out}K^2)\) | same | same | activations \(\Theta(BHWC_{out})\) |
| Max pooling | \(\Theta(BHWC)\) | same | same | output activations |
| Dense layer | \(\Theta(BdC_{out})\) | same | same | parameters plus batch activations |
| Softmax over seven logits | \(\Theta(7B)\) | same | same | \(\Theta(7B)\) |
| Reference confusion matrix | \(\Theta(N+K^2)\) | same | same | \(\Theta(K^2)\) |

EmotionCNN contains two 3x3 convolution blocks, each followed by 2x2 pooling, then dense layers of dimensions 64x12x12 to 128 to 7. Convolutional activation memory dominates small-batch inference, while the first dense layer has a substantial fixed parameter block. The reference metric implementation is \(\Theta(N+K^2)\) for \(K=7\).

Asymptotic notation is not a latency measurement. Hardware, decoding, transfer, batch size, kernel implementation, and model loading can dominate runtime. Follow docs/BENCHMARKING.md and record hardware and model checksum before making empirical claims.
