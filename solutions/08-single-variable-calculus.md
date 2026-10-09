# Chapter 8 — Solutions

## Solution 1

Claim: $\lim_{x \to 2} (2x + 1) = 5$.

Let $\varepsilon > 0$ be given. We need $\delta > 0$ such that $0 < |x - 2| < \delta \Rightarrow |(2x+1) - 5| < \varepsilon$.

i) Simplify the output distance: $|(2x + 1) - 5| = |2x - 4| = 2|x - 2|$.
ii) We want $2|x - 2| < \varepsilon$, i.e. $|x - 2| < \varepsilon/2$.
iii) Choose $\delta = \varepsilon/2 > 0$. Then $0 < |x - 2| < \delta$ gives $|(2x+1) - 5| = 2|x - 2| < 2\delta = \varepsilon$. ∎

**Basically, ...** The output distance is just twice the input distance, so the window is half the tolerance.

## Solution 2

This is the sign function.

i) For $x > 0$ (however close to $0$), $f(x) = 1$: $\lim_{x \to 0^+} f(x) = 1$.
ii) For $x < 0$ (however close to $0$), $f(x) = -1$: $\lim_{x \to 0^-} f(x) = -1$.
iii) $1 \ne -1$: the one-sided limits disagree, so $\lim_{x \to 0} f(x)$ does **not** exist.
iv) Continuity at $0$ requires the limit to equal $f(0) = 0$ — it fails already at existence. Verdict: *not continuous at $0$* (a jump discontinuity).

## Solution 3

Divide numerator and denominator by the highest power, $x^2$:
$$\lim_{x \to \infty} \frac{3x^2 + 1}{2x^2 - x} = \lim_{x \to \infty} \frac{3 + \frac{1}{x^2}}{2 - \frac{1}{x}}.$$
As $x \to \infty$, $\frac{1}{x^2} \to 0$ and $\frac{1}{x} \to 0$ (eg 3 of the chapter). The denominator tends to $2 \ne 0$, so the quotient rule for limits applies:
$$= \frac{3 + 0}{2 - 0} = \frac{3}{2}.$$

**Basically, ...** At infinity, only the leading terms survive: $\frac{3x^2}{2x^2} = \frac{3}{2}$.

## Solution 4

For $x \ne 3$:
$$\frac{x^2 - 9}{x - 3} = \frac{(x - 3)(x + 3)}{x - 3} = x + 3.$$
So $\lim_{x \to 3^-} f(x) = \lim_{x \to 3^+} f(x) = 3 + 3 = 6$ — the hole at $x = 3$ does not affect the limit. Since $f(3) = 6$ is defined to equal that limit, all three continuity conditions hold. Verdict: *continuous at $x = 3$* (the value was chosen precisely to "fill the hole").

## Solution 5

i) $x^2$ is a polynomial, hence continuous at every real $x$; $e^u$ is continuous at every real $u$, in particular at $u = x^2$. By the composition part of the continuity theorems, $e^{x^2}$ is continuous at every $x$.
ii) $\cos x$ is continuous at every real $x$.
iii) By the product part of the theorems, $e^{x^2} \cdot \cos x$ is continuous at every real $x$. No $\varepsilon$–$\delta$ argument needed — the theorems do the work.

## Solution 6

$$f'(x) = \lim_{h \to 0} \frac{\frac{1}{x+h} - \frac{1}{x}}{h}.$$

i) Combine the numerator: $\frac{1}{x+h} - \frac{1}{x} = \frac{x - (x + h)}{x(x+h)} = \frac{-h}{x(x+h)}$.
ii) Divide by $h$: $\frac{-h}{x(x+h) \cdot h} = \frac{-1}{x(x+h)}$ (valid for $h \ne 0$).
iii) Take the limit: $\lim_{h \to 0} \frac{-1}{x(x+h)} = \frac{-1}{x \cdot x} = -\frac{1}{x^2}$.

So $\left(\frac{1}{x}\right)' = -\frac{1}{x^2}$ for $x \ne 0$ — consistent with the power rule for $n = -1$.

## Solution 7

(a) $x^2 e^x$ — **product rule**. With $f = x^2$, $g = e^x$ ($f' = 2x$, $g' = e^x$):
$$\frac{d}{dx}\bigl(x^2 e^x\bigr) = 2x \cdot e^x + x^2 \cdot e^x = xe^x(x + 2).$$

(b) $\dfrac{x}{x^2 + 1}$ — **quotient rule**. With $f = x$, $g = x^2 + 1$ ($f' = 1$, $g' = 2x$; note $g(x) \ge 1 \ne 0$ always):
$$\frac{d}{dx}\left(\frac{x}{x^2+1}\right) = \frac{1 \cdot (x^2 + 1) - x \cdot 2x}{(x^2+1)^2} = \frac{1 - x^2}{(x^2+1)^2}.$$

## Solution 8

$h(x) = (2x^3 - 1)^4$: outer $f(u) = u^4$, inner $g(x) = 2x^3 - 1$.
$f'(u) = 4u^3$, $g'(x) = 6x^2$. Chain rule:
$$h'(x) = f'(g(x)) \cdot g'(x) = 4(2x^3 - 1)^3 \cdot 6x^2 = 24x^2(2x^3 - 1)^3.$$

**Basically, ...** Bring down the $4$, keep the inside untouched, then multiply by the derivative of the inside.

## Solution 9

$f(x) = x^3 - 2x$ at $a = 2$.

i) $f(2) = 8 - 4 = 4$ — the touch point is $(2, 4)$.
ii) $f'(x) = 3x^2 - 2$, so $f'(2) = 12 - 2 = 10$ — the slope.
iii) Tangent: $y = f(2) + f'(2)(x - 2) = 4 + 10(x - 2) = 10x - 16$.

## Solution 10

$\sqrt{1.02} = (1 + 0.02)^{1/2}$: take $r = \frac{1}{2}$, $x = 0.02$ (close to $0$, so the linear approximation applies):
$$(1+x)^r \approx 1 + rx = 1 + \tfrac{1}{2}(0.02) = 1.01.$$
True value: $\sqrt{1.02} \approx 1.00995049\ldots$, i.e. $1.0100$ to $4$ decimal places.
Absolute error: $|1.01 - 1.00995049\ldots| \approx 0.0000495 \approx 5 \times 10^{-5}$ — tiny, because $0.02$ is close to the expansion point $0$.

## Solution 11

$f(x) = x^3 + x^2 - x + 5$.

i) $f'(x) = 3x^2 + 2x - 1$. Set $= 0$: $3x^2 + 2x - 1 = 0$. Discriminant $4 + 12 = 16$; roots $x = \frac{-2 \pm 4}{6}$, i.e. $x = -1$ and $x = \frac{1}{3}$. Both are critical points ($f$ is a polynomial, differentiable everywhere).
ii) $f''(x) = 6x + 2$. At $x = -1$: $f''(-1) = -4 < 0 \Rightarrow$ **local maximum**, value $f(-1) = -1 + 1 + 1 + 5 = 6$. At $x = \frac{1}{3}$: $f''\!\left(\frac{1}{3}\right) = 4 > 0 \Rightarrow$ **local minimum**, value $f\!\left(\frac{1}{3}\right) = \frac{1}{27} + \frac{1}{9} - \frac{1}{3} + 5 = \frac{130}{27} \approx 4.815$.

## Solution 12

$f(x) = x^3 - 3x$ on $[-2, 2]$.

i) $f'(x) = 3x^2 - 3 = 3(x^2 - 1) = 0$ gives critical points $x = 1$ and $x = -1$, both inside $[-2, 2]$.
ii) Evaluate $f$ at the critical points and the boundary points:
$$f(-2) = -8 + 6 = -2, \quad f(-1) = -1 + 3 = 2, \quad f(1) = 1 - 3 = -2, \quad f(2) = 8 - 6 = 2.$$
iii) Largest value $2$ (at $x = -1$ and $x = 2$); smallest value $-2$ (at $x = -2$ and $x = 1$). So: **global maximum $2$**, **global minimum $-2$** — each attained twice, once interior and once at the boundary.
