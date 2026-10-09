# Solutions — Chapter 9: Multivariable calculus

## Solution 1
$f(x, y) = 3x^2y - 5xy^3 + 2x$.

$\frac{\partial f}{\partial x}$ — freeze $y$ (treat $y$ as a constant), differentiate term by term:
$$\frac{\partial}{\partial x}(3x^2y) = 3y \cdot 2x = 6xy, \qquad \frac{\partial}{\partial x}(-5xy^3) = -5y^3 \cdot 1 = -5y^3, \qquad \frac{\partial}{\partial x}(2x) = 2.$$
$$\frac{\partial f}{\partial x} = 6xy - 5y^3 + 2.$$

$\frac{\partial f}{\partial y}$ — freeze $x$:
$$\frac{\partial}{\partial y}(3x^2y) = 3x^2 \cdot 1 = 3x^2, \qquad \frac{\partial}{\partial y}(-5xy^3) = -5x \cdot 3y^2 = -15xy^2, \qquad \frac{\partial}{\partial y}(2x) = 0.$$
$$\frac{\partial f}{\partial y} = 3x^2 - 15xy^2.$$

## Solution 2
$f(x, y, z) = xy + yz + zx$.

i) $\frac{\partial f}{\partial x}$: freeze $y, z$. $\frac{\partial}{\partial x}(xy) = y$, $\frac{\partial}{\partial x}(yz) = 0$, $\frac{\partial}{\partial x}(zx) = z$. So $\frac{\partial f}{\partial x} = y + z$.
ii) $\frac{\partial f}{\partial y}$: freeze $x, z$. $\frac{\partial}{\partial y}(xy) = x$, $\frac{\partial}{\partial y}(yz) = z$, $\frac{\partial}{\partial y}(zx) = 0$. So $\frac{\partial f}{\partial y} = x + z$.
iii) $\frac{\partial f}{\partial z}$: freeze $x, y$. $\frac{\partial}{\partial z}(xy) = 0$, $\frac{\partial}{\partial z}(yz) = y$, $\frac{\partial}{\partial z}(zx) = x$. So $\frac{\partial f}{\partial z} = x + y$.

$$\nabla f = \begin{pmatrix} y + z \\ x + z \\ x + y \end{pmatrix}.$$

## Solution 3
$f(x_1, x_2) = \sin x_1 \cos x_2$.

i) $\frac{\partial f}{\partial x_1} = \cos x_1 \cos x_2$ (differentiate $\sin x_1$, freeze $\cos x_2$).
ii) $\frac{\partial f}{\partial x_2} = -\sin x_1 \sin x_2$ ($\sin x_1$ frozen; derivative of $\cos x_2$ is $-\sin x_2$).

At $\left(\frac{\pi}{2}, 0\right)$:
$$\frac{\partial f}{\partial x_1} = \cos\frac{\pi}{2} \cdot \cos 0 = 0 \cdot 1 = 0, \qquad \frac{\partial f}{\partial x_2} = -\sin\frac{\pi}{2} \cdot \sin 0 = -1 \cdot 0 = 0.$$
$$\nabla f\left(\tfrac{\pi}{2}, 0\right) = \begin{pmatrix} 0 \\ 0 \end{pmatrix}.$$
By §9.9 this is a **critical point** — a candidate for a max/min/saddle, not a verdict (the Hessian would have to interrogate it).

## Solution 4
$f(x, y) = x^2 - xy$ at $(2, -3)$, direction $\mathbf{u} = (4, 3)$.

i) Normalize: $\lVert \mathbf{u} \rVert = \sqrt{16 + 9} = 5$, so $\hat{\mathbf{u}} = \left(\frac{4}{5}, \frac{3}{5}\right) = (0.8, 0.6)$.
ii) $\nabla f(x, y) = (2x - y,\ -x)^T$ (eg 3). At $(2, -3)$: $\nabla f(2, -3) = (2(2) - (-3),\ -2)^T = (7, -2)^T$.
iii) $$D_{\hat{\mathbf{u}}} f(2, -3) = (7, -2) \cdot (0.8, 0.6) = 7(0.8) + (-2)(0.6) = 5.6 - 1.2 = 4.4.$$
(Compare eg 7: along $(0.6, 0.8)$ the rate was $2.6$; along $(0.8, 0.6)$ it is $4.4$ — different directions, different rates, same formula.)

## Solution 5
$f(x, y) = xy + y^2$ at $(1, 2)$.

i) $f(1, 2) = 1 \cdot 2 + 2^2 = 2 + 4 = 6$.
ii) $\frac{\partial f}{\partial x} = y$, so at $(1, 2)$: $2$. $\frac{\partial f}{\partial y} = x + 2y$, so at $(1, 2)$: $1 + 4 = 5$.
iii) $$L(x, y) = 6 + 2(x - 1) + 5(y - 2) = 6 + 2x - 2 + 5y - 10 = 2x + 5y - 6.$$
iv) $f(1.05, 1.9) \approx L(1.05, 1.9) = 2(1.05) + 5(1.9) - 6 = 2.1 + 9.5 - 6 = 5.6$.
Check: true value $= 1.05 \times 1.9 + 1.9^2 = 1.995 + 3.61 = 5.605$. Error $0.005$. ✓

## Solution 6
$z = e^{xy}$, $x = t^2$, $y = 1 - t$.

*Chain rule:*
$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}.$$
$\frac{\partial z}{\partial x} = y e^{xy}$, $\frac{dx}{dt} = 2t$; $\frac{\partial z}{\partial y} = x e^{xy}$, $\frac{dy}{dt} = -1$. So
$$\frac{dz}{dt} = y e^{xy} \cdot 2t + x e^{xy} \cdot (-1) = e^{xy}(2ty - x) = e^{t^2(1-t)}\bigl(2t(1-t) - t^2\bigr).$$

*Verify by substitution:*
$z(t) = e^{t^2(1-t)} = e^{t^2 - t^3}$, so $\frac{dz}{dt} = (2t - 3t^2)\, e^{t^2 - t^3}$.
And $2t(1-t) - t^2 = 2t - 2t^2 - t^2 = 2t - 3t^2$. ✓ The two routes agree.

## Solution 7
$f(x, y) = 3x + 4y - x^2 - y^2$ at $(1, 1)$.

i) $\nabla f(x, y) = (3 - 2x,\ 4 - 2y)^T$. At $(1, 1)$: $\nabla f(1, 1) = (1, 2)^T$.
ii) $\lVert \nabla f \rVert = \sqrt{1 + 4} = \sqrt{5}$.
iii) Unit steepest-ascent direction: $\dfrac{(1, 2)}{\sqrt{5}}$. Maximum rate of increase: $\lVert \nabla f \rVert = \sqrt{5} \approx 2.236$.
iv) Steepest-descent direction: $-\dfrac{(1, 2)}{\sqrt{5}} = \dfrac{(-1, -2)}{\sqrt{5}}$.

## Solution 8
$f(x, y) = x^3 + y^2 + xy$.

i) First partials: $\frac{\partial f}{\partial x} = 3x^2 + y$; $\frac{\partial f}{\partial y} = 2y + x$.
ii) Second partials: $\frac{\partial^2 f}{\partial x^2} = 6x$; $\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}(2y + x) = 1$; $\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}(3x^2 + y) = 1$; $\frac{\partial^2 f}{\partial y^2} = 2$.
iii) At $(1, 2)$:
$$\mathbf{H}(1, 2) = \begin{pmatrix} 6 & 1 \\ 1 & 2 \end{pmatrix}.$$
Yes — symmetric ($\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x} = 1$).

## Solution 9
$f(x, y) = e^{x+y}$ around $(0, 0)$.

i) $f(0, 0) = e^0 = 1$.
ii) $\frac{\partial f}{\partial x} = e^{x+y}$, $\frac{\partial f}{\partial y} = e^{x+y}$, so $\nabla f(0, 0) = (1, 1)^T$.
iii) All second partials are $e^{x+y}$, so $\mathbf{H}(0, 0) = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$.
iv) With $\mathbf{u} = (x, y)^T$:
$$\tfrac{1}{2}\, \mathbf{u}^T \mathbf{H} \mathbf{u} = \tfrac{1}{2}\,(x^2 + 2xy + y^2) = \tfrac{1}{2}(x + y)^2.$$
v) $$Q(x, y) = 1 + x + y + \tfrac{1}{2}(x + y)^2.$$
Sanity check: expand $e^{x+y}$ as a series in $s = x + y$: $e^s = 1 + s + \tfrac{1}{2}s^2 + \cdots$ — matches. ✓

## Solution 10
$f(x, y) = x^2 + 4y^2$ at $(1, 1)$.

(a) $\nabla f(x, y) = (2x,\ 8y)^T$, so $\nabla f(1, 1) = (2, 8)^T$.

(b) Level value: $f(1, 1) = 1 + 4 = 5$, so the level set is $x^2 + 4y^2 = 5$. A tangent direction: implicit differentiation gives $2x + 8y\,y' = 0$, so $y' = -\dfrac{x}{4y} = -\dfrac{1}{4}$ at $(1, 1)$. Take the tangent vector $(4, -1)$ (run $4$, rise $-1$). Then
$$\nabla f(1, 1) \cdot (4, -1) = 2 \cdot 4 + 8 \cdot (-1) = 8 - 8 = 0.$$
Zero dot product ⟹ perpendicular. ✓ (This is §9.5 in action.)

## Solution 11
$f(x, y) = x^2 + y^2 - 2x + 4y$.

$\frac{\partial f}{\partial x} = 2x - 2$; $\frac{\partial f}{\partial y} = 2y + 4$. Set both to zero:
$$2x - 2 = 0 \;\Rightarrow\; x = 1, \qquad 2y + 4 = 0 \;\Rightarrow\; y = -2.$$
The unique critical point is $(1, -2)$. (Completing the square: $f = (x-1)^2 + (y+2)^2 - 5$, so this point is in fact the global minimum — but §9.9 alone only certifies it as a candidate; the Hessian verdict waits in Chapter 10.)

## Solution 12
$$D_{\mathbf{e}_1} f(\mathbf{v}) = \nabla f(\mathbf{v}) \cdot \mathbf{e}_1 = \begin{pmatrix} \frac{\partial f}{\partial x_1}(\mathbf{v}) \\ \vdots \\ \frac{\partial f}{\partial x_d}(\mathbf{v}) \end{pmatrix} \cdot \begin{pmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix} = \frac{\partial f}{\partial x_1}(\mathbf{v}) \cdot 1 + 0 + \cdots + 0.$$
Hence $D_{\mathbf{e}_1} f(\mathbf{v}) = \frac{\partial f}{\partial x_1}(\mathbf{v})$. The directional derivative along a coordinate axis *is* the corresponding partial derivative — the partials are just the directional derivatives in the axis directions. ∎
