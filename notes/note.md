# Notes

## Broadcasting

when broadcasting `b` to `a` the shape of `b` has to "match" the shape of `a`.

### Condition 1:

$\displaystyle \mathrm{rank}_a \ge \mathrm{rank}_b$

### Condition 2:

the last $\mathrm{rank}_b$ dims of `a` have to be element-wise divisible by `b`'s dims

> $\mathrm{dims}_a \coloneqq (..., a \times n, b \times m, c \times k)$
> $\mathrm{dims}_b \coloneqq (a \times x, b \times y, c \times z)$
> with $a, b, c, n, m, k, x, y, z \in \mathbb{Z^+} : (n\mid x) \land (m \mid y) \land (k \mid z)$

for element wise operations the index used to get the value from `b` gets modulo-ed by the dim of `b`
meaning in the above example you would get each value in `b`'s $dim_0$ $n \div x$ times, $dim_1$ $m \div y$ times, $dim_2$ $k \div z$ times.

## Backprop

> change weight _and/or?_ bias tensors based on their gradient by using gradient descent (to decrease the gradient?)
> learning rate changes how much you change the tensors?

### Questions

- what exactly does gradient descent mean in a mathematical sense?
- how do you track which/calculate how much tensors should change?

## Generic Dispatch

have a generic function you pass a function to that does the element-wise (or similar) operation on the tensor(s)

### Pros

relatively easy to reason about, debug and understand what is happening
"simple" solution that works (I guess 🤷‍♂️)

### Cons
virtual call -> overhead for every op

example: addition
> addition takes one instruction per element (when not using SIMD) to process
> creating a stack frame is obviously the bottleneck by a gigantic margin

### Solution: Meta Programming To Success

macro calling a macro with all info for all ops
the called macro takes the info needed and generates a function for each op from the same generic template

maybe don't make it too generic?
I don't like the convention with passing arguments in an array
maybe have multiple kinds for unary, binary, reduce, … ops with specific parameters
