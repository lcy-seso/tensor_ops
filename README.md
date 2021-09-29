
<!-- TOC -->

- [Apply to Each](#apply-to-each)
  - [Basic mathematical operations](#basic-mathematical-operations)
  - [Logical operations](#logical-operations)
  - [Neural network-specific activations](#neural-network-specific-activations)
  - [Filter-liker operations](#filter-liker-operations)
  - [Constant / TensorCreation](#constant--tensorcreation)
- [A combination of ApplytoEach and Aggregate](#a-combination-of-applytoeach-and-aggregate)
  - [A reduce nested with a map](#a-reduce-nested-with-a-map)
  - [Interleaved ApplytoEach and reduce](#interleaved-applytoeach-and-reduce)
    - [Noramlization and Softmax](#noramlization-and-softmax)
    - [Contraction](#contraction)
- [Data access & movement](#data-access--movement)
  - [Access primitives](#access-primitives)
  - [Extended Access pattern](#extended-access-pattern)
  - [Operator with complicated access](#operator-with-complicated-access)
- [Info](#info)
  - [Modify meta information](#modify-meta-information)
  - [Query information](#query-information)
- [Ignored](#ignored)
  - [Deprecated by ONNX](#deprecated-by-onnx)
  - [Non-differentiable](#non-differentiable)
  - [Monolithic](#monolithic)
  - [Controlflow decision as an operator (high-order function)](#controlflow-decision-as-an-operator-high-order-function)
  - [Unify into TensorArray](#unify-into-tensorarray)
  - [Miscellany & Task-specific](#miscellany--task-specific)

<!-- /TOC -->

***Arity is counted as operands that has a type of tensor in the operator***. Parameters except tensors are "attributes".

# Apply to Each

All operands and the output tensor has the same access function.

## Basic mathematical operations

|Operator|Arity|Mathmatical formula|Derivative rule||
|:--|:--|:--|:--:|:-|
|Sin|binary|$y=\text{sin}(x)$|$\frac{dy}{dx}=\text{cos}(x)$|trigonometric|
|Cos|binary|$y=\text{cos}(x)$|$\frac{dy}{dx}=-\text{sin}(x)$|trigonometric|
|Tan|binary|$y=\text{tan}(x)$|$\frac{dy}{dx}=\text{sec}^2(x)$|trigonometric|
|Asin|binary|$y=\text{asin}(x)$|$\frac{dy}{dx}=\frac{1}{\sqrt{1-x^2}}$|trigonometric|
|Acos|binary|$y=\text{acos}(x)$|$\frac{dy}{dx}=-\frac{1}{\sqrt{1-x^2}}$|trigonometric|
|Atan|binary|$y=\text{atan}(x)$|$\frac{dy}{dx}=\frac{1}{1+x^2}$|trigonometric|
|Sinh|binary|$y=\text{sin}(x)$|$\frac{dy}{dx}=\text{cosh}(x)$|hyperbolic|
|Cosh|binary|$y=\text{cosh}(x)$|$\frac{dy}{dx}=\text{sinh}(x)$|hyperbolic|
|Tanh|binary|$y=\text{tanh}(x)$|$\frac{dy}{dx}=1-\text{tanh}^2(x)$|hyperbolic|
|Asinh|binary|$y=\text{sinh}(x)$|$\frac{dy}{dx}=\frac{1}{\sqrt{x^2+1}}$|inverse hyperbolic|
|Acosh|binary|$y=\text{acosh}(x)$|$\frac{dy}{dx}=\frac{1}{\sqrt{x^2-1}}$|inverse hyperbolic|
|Atanh|binary|$y=\text{atanh}(x)$|$\frac{dy}{dx}=\frac{1}{\sqrt{1-x^2}}$|inverse hyperbolic|
|Neg|binary|$y=-x$|$\frac{dy}{dx}=-1$
|Abs|binary|$y=\|x\|$|$\frac{dy}{dx}=\frac{\|x\|}{x}$|
|Exp|binary|$y=\text{exp}(x)$|$\frac{dz}{dx}=\text{exp}(x)$|
|Log|binary|$y=\text{ln}(x)$|$\frac{dz}{dx}=\frac{1}{x}$|
|Add|ternary|$z=x+y$|$\frac{dz}{dx}=1,\frac{dz}{dy}=1$|
|Sqrt|binary|$y=\sqrt{x}$|$\frac{dy}{dx}=\frac{1}{2}x^{-\frac{1}{2}}$|
|Reciprocal|binary|$y=\frac{1}{x}$|$\frac{dy}{dx}=-\frac{1}{x^2}$|
|Clip|binary|$y=\text{clip}(x, a, b)$|$\frac{dy}{dx}=0$ if $x<a$ or $x>b$; else $\frac{dy}{dx}=1$
|Div|ternary|$z=x/y$|$\frac{dz}{dx}=\frac{1}{y};\frac{dz}{dy}=-\frac{x}{y^2}$
|Pow|ternary|$z=x^y$|$\frac{dz}{dx}=yx^{y-1}$;$\frac{dz}{dy}=x^y\text{ln}(y)$;|
|Mod|ternary|$z=x \equiv y$|$\frac{dz}{dx}=1$|the second operand is non-differentiable
|Mul|ternary|$z=x*y$|$\frac{dz}{dx}=y$;$\frac{dz}{dy}=x$;|
|Sub|ternary|$z=x-y$|$\frac{dz}{dx}=1$;$\frac{dz}{dy}=-1$;|
|Sum|ternary|$z=x+y$|$\frac{dz}{dx}=1$;$\frac{dz}{dy}=1$;|
|Det|binary|
|Erf|binary|

## Logical operations

|Operators|Arity||
|:--|:--|:--
|Greater|ternary|non-differentiable|
|Equal|ternary|non-differentiable|
|And|ternary|non-differentiable|
|Not|binary|non-differentiable|
|Or|ternary|non-differentiable|
|Xor|ternary|non-differentiable|
|Less|ternary|non-differentiable|
|LessOrEqual|ternary|non-differentiable|
|GreaterOrEqual|ternary|non-differentiable|
|IsInf|binary|non-differentiable|
|IsNaN|binary|non-differentiable|

## Neural network-specific activations

|Operator|Arity|Math formula|
|:--|:--|:--|
|Sigmoid|binary|$y=1/(1 + e^{-x})$|
|HardSigmoid|binary|$y=\max(0, \min(1, \alpha * x + \beta)$)
|Hardmax|binary|
|Softplus|binary|$y=\log(1 + e^x)$
|Softsign|binary|$y=x/(1 + \|x\|)$
|LeakyRelu|binary|
|PRelu|ternary|
|Relu|binary|$y=\max(0, x)$
|Selu|binary|$y = \gamma * (\alpha * e^x - \alpha) \text{  if  } x \le 0; y = \gamma * x \text{  if  } x > 0$|
|Elu|binary|$x \text{  if  } x \ge 0; \text{   else  } \alpha*(e^x - 1)$
|Celu|binary|$y=\max(0,x) + \min(0,\alpha*(\exp(x / \alpha) -1))$|
|ThresholdedRelu|binary|$y = x \text{  if  } x > \alpha; y = 0 \text{  otherwise}$
|HardSwish|binary|$y = x * \max(0, \min(1, \alpha * x + \beta)$
|Dropout|binary|

## Filter-liker operations

|Operators|airty|
|:--|:--|
|***NonZero***|binary|
|***Unique***|binary|
|***Where***|ternary|
|***Compress***|binary|

## Constant / TensorCreation

|Operator|Arity||
|:--|:--:|:--
|Range|unary|
|Constant|unary|
|ConstantOfShape|unary|
|EyeLike|unary|
|RandomNormalLike|unary|
|RandomUniform|unary|
|RandomUniformLike|unary|
|RandomNormal|unary
|Bernoulli|unary
|Multinomial|unary
|OneHot|binary|non-differentiable
|Pad|binary|

# A combination of ApplytoEach and Aggregate

## A reduce nested with a map

|Operator|Arity||
|:--|:--|:--|
|TopK|Binary|
|CumSum|Binary|like scan|
|ReduceL1|Binary|
|ReduceL2|Binary|
|ReduceLogSum|Binary|
|ReduceLogSumExp|Binary|
|ArgMax|binary|non-differentiable
|ArgMin|binary|non-differentiable
|ReduceMax|binary|
|ReduceMean|binary|
|ReduceMin|binary|
|ReduceProd|binary|
|ReduceSum|binary|
|ReduceSumSquare|binary|
|Max|variadic|
|Mean|variadic|
|Min|variadic|

## Interleaved ApplytoEach and reduce

### Noramlization and Softmax

|Operator|Arity|
|:--|:--|
|BatchNormalization|6|
|LpNormalization|binary|
|InstanceNormalization|4|
|LRN|binary|
|Softmax|binary|
|LogSoftmax|binary|
|MeanVarianceNormalization|binary|
|SoftmaxCrossEntropyLoss|ternary|
|NegativeLogLikelihoodLoss|4|

### Contraction

|Operator|Arity|
|:--|:--:|
|Conv|ternary|
|ConvInteger|ternary|
|ConvTranspose|ternary|
|Gemm|ternary|

# Data access & movement

## Access primitives

|Operator|Arity|
|:--|:--:|
|Concat|variadic|
|Split|variadic|
|Expand|binary|
|Transpose|binary|
|Gather|ternary|
|GatherElements|ternary|
|GatherND|ternary|
|ScatterElements|4
|ScatterND|4
|Identity|equals to copy|

## Extended Access pattern

|Operator|Arity||
|:--|:--|:--|
|DepthToSpace|binary|permutation|
|SpaceToDepth|binary|permutation|
|Slice|binary|
|Tile|binary|
|Trilu|binary|

## Operator with complicated access

|Operator|Arity|Access pattern|
|:--|:--|:--|
|AveragePool|binary|window+map|
|MaxPool|binary|window+map|
|MaxUnpool|binary|window+map|
|LpPool|binary|window+map|
|GlobalAveragePool|binary|window+map|
|GlobalLpPool|binary|window+map|
|GlobalMaxPool|binary|window+map|

# Info

## Modify meta information

|Operator|Arity|
|:--|:--|
|Flatten|binary|
|Reshape|binary|
|Resize|binary|
|Squeeze|binary|
|Unsqueeze|binary|

## Query information

|Operator|
|:--|
|Size|
|Shape|
|Shrink|

---

# Ignored

## Deprecated by ONNX

|Operator|
|:--|
|Upsample|
|Scatter|

## Non-differentiable

|Operator|Arity|
|:--|:--|
|BitShift|ternary|
|Ceil|binary|
|Floor|binary|
|Round|binary|
|Sign|binary|
|Cast|binary|
|CastLike|binary|

## Monolithic

|Operator||
|:--|:--|
|Einsum|A fusion of map, reduce and transpose|
|GRU|A network as an operator|
|LSTM|A network as an operator|
|RNN|A network as an operator|

## Controlflow decision as an operator (high-order function)

|Operator|
|:--|
|Loop|
|If|
|Scan|

## Unify into TensorArray

|Operators|
|:--|
|ConcatFromSequence|
|SequenceAt|binary|
|SequenceConstruct|
|SequenceEmpty|
|SequenceErase|
|SequenceInsert|
|SequenceLength|
|SplitToSequence|
|ReverseSequence|

## Miscellany & Task-specific

|Operator||
|:--|:--|
|Optional||
|OptionalGetElement||
|OptionalHasElement||
|StringNormalizer|
|TfIdfVectorizer|
|DynamicQuantizeLinear|
|DequantizeLinear|
|QuantizeLinear|
|QLinearConv||
|QLinearMatMul||
|NonMaxSuppression|for image detection|
|MaxRoiPool|for image detection|
|RoiAlign|for image detection|