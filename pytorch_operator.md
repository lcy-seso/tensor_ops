<!-- vscode-markdown-toc -->
- [Common](#common)
  - [Arithmetic Operations](#arithmetic-operations)
  - [Gather/Scatter & Data movements](#gatherscatter--data-movements)
  - [View/Reshape](#viewreshape)
  - [RNNs](#rnns)
  - [Factory](#factory)
  - [Linalg](#linalg)
  - [Miscellany](#miscellany)
- [Ignored](#ignored)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# Common

The operator list is from [functorch](https://github.com/facebookresearch/functorch/blob/main/op_analysis/annotated_ops.txt). Convolution operations are not included in this list.

## Arithmetic Operations

|Operators|Annotation|
|:--|:--|
|abs| primitive pointwise|
|acos| primitive pointwise|
|acosh| primitive pointwise|
|add| primitive pointwise|
|asin| primitive pointwise|
|asinh| primitive pointwise|
|atan| primitive pointwise|
|atan2| primitive pointwise|
|atanh| primitive pointwise|
|bitwise_and| primitive pointwise|
|bitwise_left_shift| primitive pointwise|
|bitwise_not| primitive pointwise|
|bitwise_or| primitive pointwise|
|bitwise_right_shift| primitive pointwise|
|bitwise_xor| primitive pointwise|
|ceil| primitive pointwise|
|cos| primitive pointwise|
|cosh| primitive pointwise|
|digamma| primitive pointwise|
|div| primitive pointwise|
|eq| primitive pointwise|
|erf| primitive pointwise|
|erfinv| primitive pointwise|
|exp| primitive pointwise|
|floor| primitive pointwise|
|fmod| primitive pointwise|
|frac| primitive pointwise|
|gcd| primitive pointwise|
|ge| primitive pointwise|
|gt| primitive pointwise|
|i0| primitive pointwise|
|igamma| primitive pointwise|
|igammac| primitive pointwise|
|le| primitive pointwise|
|lgamma| primitive pointwise|
|log| primitive pointwise|
|lt| primitive pointwise|
|mvlgamma| primitive pointwise|
|polygamma| primitive pointwise|
|pow| primitive pointwise|
|round| primitive pointwise|
|rsub| primitive pointwise|
|sgn| primitive pointwise|
|sin| primitive pointwise|
|sinh| primitive pointwise|
|special_digamma| primitive pointwise|
|special_entr| primitive pointwise|
|special_erfcx| primitive pointwise|
|special_i1| primitive pointwise|
|special_log1p| primitive pointwise|
|special_ndtr| primitive pointwise|
|special_ndtri| primitive pointwise|
|special_psi| primitive pointwise|
|special_round| primitive pointwise|
|special_sinc| primitive pointwise|
|special_xlogy| primitive pointwise|
|special_zeta| primitive pointwise|
|sqrt| primitive pointwise|
|sub| primitive pointwise|
|tan| primitive pointwise|
|tanh| primitive pointwise|
|true_divide| primitive pointwise|
|addbmm| composite matmul|
|addmm| composite matmul|
|addmv| composite matmul|
|baddbmm| composite matmul|
|bilinear| composite matmul|
|bmm| composite matmul|
|einsum| composite matmul|
|linear| composite matmul|
|matmul| composite matmul|
|mm| composite matmul|
|mv| composite matmul|
|tensordot| composite matmul|
|addcdiv| composite pointwise|
|addcmul| composite pointwise|
|addr| composite pointwise|
|angle| composite pointwise|
|celu| composite pointwise|
|clamp| composite pointwise|
|clamp_max| composite pointwise|
|clamp_min| composite pointwise|
|copysign| composite pointwise|
|deg2rad| composite pointwise|
|dropout| composite pointwise|
|elu| composite pointwise|
|erfc| composite pointwise|
|exp2| composite pointwise|
|expm1| composite pointwise|
|feature_dropout| composite pointwise|
|float_power| composite pointwise|
|floor_divide| composite pointwise|
|frexp| composite pointwise|
|gelu| composite pointwise|
|glu| composite pointwise|
|hardshrink| composite pointwise|
|hardsigmoid| composite pointwise|
|hardswish| composite pointwise|
|hardtanh| composite pointwise|
|heaviside| composite pointwise|
|hypot| composite pointwise|
|isclose| composite pointwise|
|isfinite| composite pointwise|
|isinf| composite pointwise|
|isnan| composite pointwise|
|isneginf| composite pointwise|
|isposinf| composite pointwise|
|isreal| composite pointwise|
|kron| composite pointwise|
|lcm| composite pointwise|
|ldexp| composite pointwise|
|leaky_relu| composite pointwise|
|lerp| composite pointwise|
|log10| composite pointwise|
|log1p| composite pointwise|
|log2| composite pointwise|
|log_sigmoid| composite pointwise|
|log_sigmoid_forward| composite pointwise|
|logical_and| composite pointwise|
|logical_not| composite pointwise|
|logical_or| composite pointwise|
|logical_xor| composite pointwise|
|logit| composite pointwise|
|maximum| composite pointwise|
|minimum| composite pointwise|
|mish| composite pointwise|
|nan_to_num| composite pointwise|
|ne| composite pointwise|
|neg| composite pointwise|
|nextafter| composite pointwise|
|outer| composite pointwise|
|polar| composite pointwise|
|positive| composite pointwise|
|prelu| composite pointwise|
|rad2deg| composite pointwise|
|reciprocal| composite pointwise|
|relu| composite pointwise|
|relu6| composite pointwise|
|remainder| composite pointwise|
|rrelu| composite pointwise|
|rrelu_with_noise| composite pointwise|
|rsqrt| composite pointwise|
|selu| composite pointwise|
|sigmoid| composite pointwise|
|sign| composite pointwise|
|signbit| composite pointwise|
|silu| composite pointwise|
|sinc| composite pointwise|
|softplus| composite pointwise|
|softshrink| composite pointwise|
|special_expit| composite pointwise|
|special_i0e| composite pointwise|
|special_i1e| composite pointwise|
|special_logit| composite pointwise|
|special_xlog1py| composite pointwise|
|square| composite pointwise|
|threshold| composite pointwise|
|trapz| composite pointwise|
|trunc| composite pointwise|
|xlogy| composite pointwise|
|all| reduction|
|alpha_dropout| reduction|
|amax| reduction|
|amin| reduction|
|any| reduction|
|argmax| reduction|
|argmin| reduction|
|binary_cross_entropy| reduction|
|binary_cross_entropy_with_logits| reduction|
|cdist| reduction|
|cosine_embedding_loss| reduction|
|cosine_similarity| reduction|
|count_nonzero| reduction|
|cross_entropy_loss| reduction|
|cummax| reduction|
|cummin| reduction|
|cumprod| reduction|
|cumsum| reduction|
|diff| reduction|
|dist| reduction|
|dot| reduction|
|embedding_bag| reduction|
|feature_alpha_dropout| reduction|
|fmax| reduction|
|fmin| reduction|
|frobenius_norm| reduction|
|group_norm| reduction|
|hinge_embedding_loss| reduction|
|huber_loss| reduction|
|inner| reduction|
|instance_norm| reduction|
|isin| reduction|
|kl_div| reduction|
|kthvalue| reduction|
|l1_loss| reduction|
|layer_norm| reduction|
|linalg_matrix_norm| reduction|
|linalg_norm| reduction|
|linalg_vector_norm| reduction|
|log_softmax| reduction|
|logaddexp| reduction|
|logaddexp2| reduction|
|logcumsumexp| reduction|
|logsumexp| reduction|
|margin_ranking_loss| reduction|
|max| reduction|
|mean| reduction|
|median| reduction|
|min| reduction|
|mse_loss| reduction|
|mul| reduction|
|multi_margin_loss| reduction|
|multilabel_margin_loss| reduction|
|multilabel_margin_loss_forward| reduction|
|nanmedian| reduction|
|nansum| reduction|
|nll_loss| reduction|
|nll_loss2d| reduction|
|nll_loss2d_forward| reduction|
|nll_loss_forward| reduction|
|nll_loss_nd| reduction|
|norm| reduction|
|norm_except_dim| reduction|
|nuclear_norm| reduction|
|pairwise_distance| reduction|
|pdist| reduction|
|poisson_nll_loss| reduction|
|prod| reduction|
|renorm| reduction|
|smooth_l1_loss| reduction|
|soft_margin_loss| reduction|
|softmax| reduction|
|std| reduction|
|std_mean| reduction|
|sum| reduction|
|sum_to_size| reduction|
|trace| reduction|
|trapezoid| reduction|
|triplet_margin_loss| reduction|
|var| reduction|
|var_mean| reduction|
|vdot| reduction|

## Gather/Scatter & Data movements

|Operator|Annotation|
|:--|:--|
|gather| scatter/gather|
|index| scatter/gather|
|index_add| scatter/gather|
|index_copy| scatter/gather|
|index_fill| scatter/gather|
|index_put| scatter/gather|
|index_select| scatter/gather|
|masked_select| scatter/gather|
|one_hot| scatter/gather|
|permute| scatter/gather|
|put| scatter/gather|
|scatter| scatter/gather|
|scatter_add| scatter/gather|
|take| scatter/gather|
|take_along_dim| scatter/gather|

## View/Reshape

|||
|:--|:--|
|as_strided| view/reshape|
|atleast_1d| view/reshape|
|atleast_2d| view/reshape|
|atleast_3d| view/reshape|
|block_diag| view/reshape|
|broadcast_tensors| view/reshape|
|broadcast_to| view/reshape|
|cat| view/reshape|
|channel_shuffle| view/reshape|
|chunk| view/reshape|
|clone| view/reshape|
|column_stack| view/reshape|
|constant_pad_nd| view/reshape|
|contiguous| view/reshape|
|diag| view/reshape|
|diag_embed| view/reshape|
|diagflat| view/reshape|
|diagonal| view/reshape|
|dsplit| view/reshape|
|dstack| view/reshape|
|expand| view/reshape|
|expand_as| view/reshape|
|flatten| view/reshape|
|flip| view/reshape|
|fliplr| view/reshape|
|flipud| view/reshape|
|hsplit| view/reshape|
|hstack| view/reshape|
|meshgrid| view/reshape|
|movedim| view/reshape|
|narrow| view/reshape|
|narrow_copy| view/reshape|
|numpy_T| view/reshape|
|pixel_shuffle| view/reshape|
|pixel_unshuffle| view/reshape|
|ravel| view/reshape|
|repeat| view/reshape|
|repeat_interleave| view/reshape|
|reshape| view/reshape|
|reshape_as| view/reshape|
|roll| view/reshape|
|rot90| view/reshape|
|select| view/reshape|
|slice| view/reshape|
|split| view/reshape|
|split_with_sizes| view/reshape|
|squeeze| view/reshape|
|stack| view/reshape|
|t| view/reshape|
|tensor_split| view/reshape|
|tile| view/reshape|
|transpose| view/reshape|
|tril| view/reshape|
|triu| view/reshape|
|unbind| view/reshape|
|unflatten| view/reshape|
|unsafe_chunk| view/reshape|
|unsafe_split| view/reshape|
|unsafe_split_with_sizes| view/reshape|
|unsqueeze| view/reshape|
|view| view/reshape|
|view_as| view/reshape|
|vsplit| view/reshape|
|vstack| view/reshape|
|**col2im**| misc|
|**im2col**| view/reshape|

## RNNs

|Operator|Annotation|
|:--|:--|
|gru| rnn|
|gru_cell| rnn|
|lstm| rnn|
|lstm_cell| rnn|
|quantized_gru_cell| rnn|
|quantized_lstm_cell| rnn|
|quantized_rnn_relu_cell| rnn|
|quantized_rnn_tanh_cell| rnn|
|rnn_relu| rnn|
|rnn_relu_cell| rnn|
|rnn_tanh| rnn|
|rnn_tanh_cell| rnn|

## Factory

|Operator|Annotation|
|:--|:--|
|affine_grid_generator| factory|
|arange| factory|
|bartlett_window| factory|
|bernoulli| factory|
|binomial| factory|
|blackman_window| factory|
|empty| factory|
|empty_like| factory|
|empty_quantized| factory|
|empty_strided| factory|
|eye| factory|
|full| factory|
|full_like| factory|
|hamming_window| factory|
|hann_window| factory|
|kaiser_window| factory|
|linspace| factory|
|logspace| factory|
|new_empty| factory|
|new_empty_strided| factory|
|new_full| factory|
|new_ones| factory|
|new_zeros| factory|
|normal| factory|
|ones| factory|
|ones_like| factory|
|poisson| factory|
|rand| factory|
|rand_like| factory|
|randint| factory|
|randint_like| factory|
|randn| factory|
|randn_like| factory|
|randperm| factory|
|range| factory|
|scalar_tensor| factory|
|tril_indices| factory|
|triu_indices| factory|
|vander| factory|
|zeros| factory|
|zeros_like| factory|

## Linalg

|Operator|Annotation|
|:--|:--|
|cholesky| linalg|
|cholesky_inverse| linalg|
|cholesky_solve| linalg|
|eig| linalg|
|geqrf| linalg|
|inverse| linalg|
|linalg_cond| linalg|
|linalg_det| linalg|
|linalg_eigh| linalg|
|linalg_eigvalsh| linalg|
|linalg_householder_product| linalg|
|linalg_lstsq| linalg|
|linalg_matrix_power| linalg|
|linalg_multi_dot| linalg|
|linalg_svdvals| linalg|
|linalg_tensorinv| linalg|
|linalg_tensorsolve| linalg|
|logdet| linalg|
|lu_solve| linalg|
|lu_unpack| linalg|
|matrix_exp| linalg|
|matrix_rank| linalg|
|ormqr| linalg|
|pinverse| linalg|
|qr| linalg|
|slogdet| linalg|
|solve| linalg|
|svd| linalg|
|symeig| linalg|
|triangular_solve| linalg|

## Miscellany

|||
|:--|:--|
|alias| misc|
|argsort| misc|
|bincount| misc|
|bucketize| misc|
|cartesian_prod| misc|
|combinations| misc|
|cross| misc|
|ctc_loss| misc|
|data| misc|
|detach| misc|
|embedding| misc|
|flatten_dense_tensors| misc|
|from_file| misc|
|gradient| misc|
|grid_sampler| misc|
|grid_sampler_2d| misc|
|grid_sampler_3d| misc|
|histc| misc|
|masked_fill| misc|
|masked_scatter| misc|
|mode| misc|
|msort| misc|
|multinomial| misc|
|nanquantile| misc|
|nonzero| misc|
|nonzero_numpy| misc|
|pad_sequence| misc|
|pin_memory| misc|
|quantile| misc|
|reflection_pad1d| misc|
|reflection_pad2d| misc|
|replication_pad1d| misc|
|replication_pad2d| misc|
|replication_pad3d| misc|
|searchsorted| misc|
|segment_reduce| misc|
|sort| misc|
|to| misc|
|topk| misc|
|type_as| misc|
|unflatten_dense_tensors| misc|
|unfold| misc|
|unique_consecutive| misc|
|unique_dim| misc|
|unique_dim_consecutive| misc|
|upsample_bicubic2d| misc|
|upsample_bilinear2d| misc|
|upsample_linear1d| misc|
|upsample_nearest1d| misc|
|upsample_nearest2d| misc|
|upsample_nearest3d| misc|
|upsample_trilinear3d| misc|
|where| misc|

# Ignored

|Operator|Annotation|
|:--|:--|
|complex| complex|
|conj| complex|
|conj_physical| complex|
|imag| complex|
|real| complex|
|resolve_conj| complex|
|view_as_complex| complex|
|view_as_real| complex|
|absolute| alias|
|arccos| alias|
|arccosh| alias|
|arcsin| alias|
|arcsinh| alias|
|arctan| alias|
|arctanh| alias|
|chain_matmul| alias|
|clip| alias|
|det| alias|
|divide| alias|
|fix| alias|
|ger| alias|
|greater| alias|
|greater_equal| alias|
|less| alias|
|less_equal| alias|
|linalg_cholesky| alias|
|linalg_cholesky_ex| alias|
|linalg_eig| alias|
|linalg_eigvals| alias|
|linalg_inv| alias|
|linalg_inv_ex| alias|
|linalg_matrix_rank| alias|
|linalg_pinv| alias|
|linalg_qr| alias|
|linalg_slogdet| alias|
|linalg_solve| alias|
|linalg_svd| alias|
|lstsq| alias|
|matrix_power| alias|
|moveaxis| alias|
|multiply| alias|
|negative| alias|
|not_equal| alias|
|orgqr| alias|
|row_stack| alias|
|special_erf| alias|
|special_erfc| alias|
|special_erfinv| alias|
|special_exp2| alias|
|special_expm1| alias|
|special_gammaln| alias|
|special_i0| alias|
|subtract| alias|
|swapaxes| alias|
|swapdims| alias|
|choose_qparams_optimized| quantize|
|dequantize| quantize|
|fake_quantize_per_channel_affine| quantize|
|fake_quantize_per_channel_affine_cachemask| quantize|
|fake_quantize_per_tensor_affine| quantize|
|fake_quantize_per_tensor_affine_cachemask| quantize|
|int_repr| quantize|
|q_per_channel_scales| quantize|
|q_per_channel_zero_points| quantize|
|quantize_per_channel| quantize|
|quantize_per_tensor| quantize|
|coalesce| sparse|
|col_indices| sparse|
|crow_indices| sparse|
|hspmm| sparse|
|indices| sparse|
|smm| sparse|
|sparse_coo_tensor| sparse|
|sparse_csr_tensor| sparse|
|sparse_mask| sparse|
|sspaddmm| sparse|
|to_dense| sparse|
|to_sparse| sparse|
|values| sparse|
|align_as| named|
|align_tensors| named|
|align_to| named|
|refine_names| named|
|rename| named|
|fbgemm_linear_fp16_weight| fbgemm|
|fbgemm_linear_fp16_weight_fp32_activation| fbgemm|
|fbgemm_linear_int8_weight| fbgemm|
|fbgemm_linear_int8_weight_fp32_activation| fbgemm|
|fbgemm_linear_quantize_weight| fbgemm|
|fbgemm_pack_gemm_matrix_fp16| fbgemm|
|fbgemm_pack_quantized_matrix| fbgemm|
|fft_fft| fft|
|fft_fft2| fft|
|fft_fftfreq| fft|
|fft_fftn| fft|
|fft_fftshift| fft|
|fft_hfft| fft|
|fft_ifft| fft|
|fft_ifft2| fft|
|fft_ifftn| fft|
|fft_ifftshift| fft|
|fft_ihfft| fft|
|fft_irfft| fft|
|fft_irfft2| fft|
|fft_irfftn| fft|
|fft_rfft| fft|
|fft_rfft2| fft|
|fft_rfftfreq| fft|
|fft_rfftn| fft|
|istft| fft|
|stft| fft|
