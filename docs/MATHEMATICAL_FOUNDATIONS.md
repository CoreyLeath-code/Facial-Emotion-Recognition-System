# Mathematical foundations

## Implemented classification function

For an input grayscale image \(x \in [0,1]^{1 \times 48 \times 48}\), the supported PyTorch CNN maps convolution, ReLU, max-pooling, flattening, and two dense layers to logits

\[
z=f_\theta(x)\in\mathbb{R}^{7}.
\]

The seven coordinates correspond to the label map in src/src/modeling/model.py. A logit is an unnormalized score; EmotionCNN.forward does not apply softmax.

At inference, probabilities are computed as

\[
p_k = \frac{\exp(z_k-m)}{\sum_{j=1}^{7}\exp(z_j-m)}, \qquad m=\max_j z_j.
\]

Subtracting \(m\) leaves the distribution unchanged while reducing exponential overflow risk. PyTorch performs this calculation in EmotionModel.predict; src/research/classification_reference.py is an independent educational implementation with analytical tests.

The predicted expression label is \(\arg\max_k p_k\). It is a class assignment over the training labels, not an assertion about an internal emotional state.

## Training objective represented by the code

For a one-hot class target \(y\), categorical cross entropy is

\[
\mathcal{L}(z,y)=-\sum_{k=1}^{7}y_k\log p_k.
\]

The legacy TensorFlow training script in train.py configures categorical cross entropy and Adam. It is documented because it exists in the repository, but it produces a different framework artifact from the supported PyTorch API and is not a validated training route for that API.

For an integer target \(t\), the reference implementation uses equivalent loss \(-\log p_t\). Its tests cover normalization, logit-shift invariance, an analytical uniform case, and invalid inputs.

## Evaluation definitions

Let \(C_{ij}\) count examples with true class \(i\) predicted as class \(j\). Then

\[
\operatorname{accuracy}=\frac{\sum_i C_{ii}}{\sum_{i,j}C_{ij}},\quad
P_i=\frac{C_{ii}}{\sum_j C_{ji}},\quad
R_i=\frac{C_{ii}}{\sum_j C_{ij}}.
\]

Per-class F1 is \(F1_i=2P_iR_i/(P_i+R_i)\) when its denominator is nonzero, and macro F1 averages the seven classwise values. The reference summary defines zero-support divisions as zero. No repository artifact currently supplies measured values for these quantities.
