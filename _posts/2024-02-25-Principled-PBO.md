---
layout: post
title: Principled PBO
date: 2025-12-31 00:00:00-0400
description:
tags: Bayesian-optimization
categories: machine-learning-posts
---

### Problem Statement

$$
\begin{equation}
\end{equation}
$$

$$
\begin{eqnarray}
\ell_t(\hat{f}) &=& \log \mathbb{P}_{\hat{f}}((x_\tau, x^\prime_\tau, \mathbf{1}_\tau)_{\tau = 1}^t) = \sum_{\tau = 1}^t \log p_{\hat{f}}(x_\tau, x^\prime_\tau, \mathbf{1}_\tau) \\
&=& \sum_{\tau = 1}^t (z_\tau \mathbf{1}_\tau + z^\prime_\tau(1 - \mathbf{1}_\tau)) - \sum_{\tau = 1}^t \log(e^{z_\tau} + e^{z^\prime_\tau}) \\
\end{eqnarray}
$$

Maximum likelihood estimator (MLE):

$$
\hat{f}_t^{\mathrm{MLE}} \in \underset{\tilde{f} \in \mathcal{B}_f}{\arg \max} \log \mathbb{P}_{\hat{f}}((x_\tau, x^\prime_\tau, \mathbf{1}_\tau)_{\tau = 1}^t)
$$

**Theorem 3.1 (Likelihood-based confidence set)**

*$$\forall \epsilon, \delta > 0$$, let,*

$$
\mathcal{B}_f^{t + 1} = \{ \tilde{f} \in \mathcal{B}_f \vert \ell_t(\tilde{f}) \geq \ell_t(\hat{f}_t^\mathrm{MLE}) - \beta_1(\epsilon, \delta, t) \}
$$

*where $$\beta_1(\epsilon, \delta, t) = \sqrt{32 t B^2 \log \frac{ \pi^2 t^2 \mathcal{N}(\mathcal{B}_f, \epsilon, \Vert . \Vert_\infty) }{6 \delta}} + C_{L \epsilon t} = \mathcal{O}\left( \sqrt{t \log \frac{t \mathcal{N}(\mathcal{B}_f, \epsilon, \Vert . \Vert_\infty)}{\delta}} \right)$$, with $$C_L$$ a constant independent of $$\delta, t$$, and $$\epsilon$$. We have,*

$$
\mathbb{P}(f \in \mathcal{B}_f^{t + 1}, \forall t \geq 1) \geq 1 - \delta
$$

Pointwise confidence range for the black-box function

$$
\underset{\tilde{f} \in \mathcal{B}_{f}^t}{\inf} \tilde{f}(x) \leq f(x) \leq \underset{\tilde{f} \in \mathcal{B}_f^t}{\sup} \tilde{f}(x)
$$

**Theorem 3.6 (Duel-wise error bound)**

*For any estimate $$\hat{f}_{t + 1} \in \mathcal{B}_f^{t + 1}$$ measurable w.r.t. $$\mathcal{F}_t$$, we have, with probability at least $$1 - \delta, \forall t \geq 1, (x, x^\prime) \in \mathcal{X} \times \mathcal{X}$$,*

$$
\vert (\hat{f}_{t + 1}(x) - \hat{f}_{t + 1}(x^\prime)) - (f(x) - f(x^\prime)) \vert \leq 2(2B + \lambda^{-1/2} \sqrt{\beta(\epsilon, \delta / 2, t)}) \sigma_{t + 1}^{f f^\prime}((x, x^\prime))
$$

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/POPBO-algorithm.png" class="img-fluid rounded z-depth-1"  zoomable=true style="max-width: 25%;" %}
    </div>
</div>
<div class="caption">
    The algorithm of principled optimistic preferential BO.
</div>

By the representer theorem, the maximum likelihood estimation problem can solved through

$$
\begin{eqnarray}
&&\ell_t(\hat{f}_t^\mathrm{MLE}) = \underset{Z_{0:t \in \mathbb{R}^{t + 1}}}{\max} \ell(Z_{0:t} \vert \mathcal{D}_t) \nonumber \\
&&\text{subject to} \; Z_{0:t}^\top K_{0:t}^{-1} Z_{0:t} \leq B^2, 
\end{eqnarray}
$$

where $$K_{0:t} = (k(x_{\tau_1}, x_{\tau_2}))_{\tau_1 \in {0} \cup [t], \tau_2 \in \{ 0 \} \cup [t] }$$.

$$
\begin{eqnarray}
&&\underset{\tilde{f}}{\max} \tilde{f}(x) - \tilde{f}(x_t) \nonumber \\
&&\text{subject to} \tilde{f} \in \mathcal{B}_f; \ell_t(\hat{f}_t^\mathrm{MLE}) - \beta_1(\epsilon, \delta, t),
\end{eqnarray}
$$

