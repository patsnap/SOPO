这是一个极其精炼且直击要害的版本。去除了长篇大论，将重心完全聚焦在**核心机制（MoE + 动态采样）如何从根本上扭转算力开销**，非常适合直接用作 Rebuttal 的正式回复。

---

### Computational Cost Analysis (Response to Rev. ts6K, Tm3a, NHiH)

To rigorously address concerns regarding the computational overhead of the 80B Teacher, we formulate the exact training FLOPs using standard scaling laws ($2ND$ for forward generation, $6ND$ for backward update). Let $T$ be the total trajectories, and $\rho$ be the dynamic sampling ratio.

*   **GRPO (Baseline):** Updates on 100% of trajectories. $\mathcal{F}_{GRPO} = 8 N_{stu} D_{grpo} T$.
*   **SOPO (Ours):** Separates generation (100% of $T$) and update ($\rho$ fraction of $T$). Importantly, our Qwen3-Next-80B-A3B Teacher is an MoE architecture with only **8B active parameters** ($N_{tea} = 8B$).
    $$ \mathcal{F}_{SOPO} = \underbrace{2N_{stu}D_{stu\_gen}T + 2N_{tea}D_{tea\_in}T}_{\text{Forward: 100\% } T} + \underbrace{6N_{stu}D_{sopo}(\rho \cdot T)}_{\text{Backward: Only } \rho \text{ of } T} $$

Based on our exact training logs ($T=320,000$, $\rho=0.4802$), we present the precise FLOPs breakdown. *(Note: $D$ includes the substantial system prompts processed by both models).*

**Table: Detailed Training FLOPs Breakdown (in ExaFLOPs, $10^{18}$)**

| Training Phase & Compute | GRPO (Baseline) | SOPO (w/o Dynamic Sampling) | **SOPO (Full)** |
| :--- | :--- | :--- | :--- |
| **1. Forward Gen ($2ND$)** |
| Student Model (100% $T$) | 7.58 | 8.43 | **8.43** |
| Teacher Model (100% $T$) | 0.00 | 7.13 | **7.13** |
| **2. Backward Update ($6ND$)** |
| Student Model | 22.73 *(100% $T$)* | 26.80 *(100% $T$)* | **12.87** *(48.02% $T$)* |
| **3. Total Epoch Compute** | **30.31** | **42.36** | **28.43** |
| *Relative Cost vs. GRPO* | *-* | *(+39.7%)* | **(-6.2%)** |

**Key Takeaways:**
1. **The MoE Teacher is Lightweight:** Even when processing 100% of the trajectories, the 8B active Teacher only consumes 7.13 EFLOPs. The dominant bottleneck is the $6ND$ backward pass of the student.
2. **Dynamic Sampling is the Key to Efficiency:** Forcing SOPO to backpropagate through all multi-turn trajectories ($\rho=1.0$) would increase compute by $39.7\%$. However, Socratic feedback rapidly drives the policy toward high confidence. By dynamically dropping zero-variance groups ($\rho=0.4802$), **SOPO completely offsets the Teacher's overhead, making a full training run mathematically cheaper than the baseline (-6.2%).**
3. **Iso-Performance Savings:** Furthermore, because dense Socratic guidance accelerates convergence, SOPO reaches the baseline's peak performance using only $\sim50\%$ of the training steps. Consequently, achieving the state-of-the-art target accuracy requires only **$\sim14.2$ EFLOPs**, representing a massive **$>50\%$ net reduction** in total computational cost. We will include this analysis in the revised Limitation and Efficiency section.