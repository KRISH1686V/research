# Reading budget prices from dV/db of a budget-conditioned critic: novelty check

**Bottom line.** The theory is standard, so it cannot be claimed as new. I found no paper that reads prices from the gradient of a budget-conditioned deep critic in place of trained multipliers, and none that compares such a price with trained Lagrangian multipliers. I found this neither on robots nor with rolling budgets. That is a search result, not proof of absence (see the search log in the last section). Three papers come close and must be cited: Boutilier & Lu 2016, Cai et al. 2017 (RLB) and Carrara et al. 2019 (BFTQ).

**How to read the evidence tags.**
- **[read]**: I read the passage in the source's own text, extracted from the PDF or a raw page.
- **[word search]**: I searched the extracted full text for terms but did not read the whole paper, so absences are weak evidence. A phrase split across lines could be missed.
- **[record]**: I saw only an abstract or bibliographic record.
- **[snippet]**: I saw only a search tool's summary, not the source. Treat these as leads.

---

## Which papers derive the link between the multiplier and the sensitivity of the optimal value, and is it standard?

### Takeaway
Yes, it is standard. The cleanest general statement is Boyd & Vandenberghe (5.58). The cleanest CMDP statement I found is Lemma 2 of Ding, Huan & Ribeiro 2024: a vector is a multiplier exactly when it is a subgradient of the negative optimal value in the relaxation. The gradient form holds only where the value is differentiable. Everywhere else, only the subgradient form holds.

### Cited Findings

**General convex theory**
- [read] Boyd & Vandenberghe §5.6.2 gives a global inequality, assuming strong duality and an attained dual optimum: p⋆(u,v) ≥ p⋆(0,0) − λ⋆ᵀu − ν⋆ᵀv (5.57). They note "This is the case if the original problem is convex, and Slater's condition is satisfied" — [Convex Optimization, pp. 250-251](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- [read] §5.6.3 gives the local form: if p⋆ is differentiable at (0,0) and strong duality holds, then λ⋆ᵢ = −∂p⋆(0,0)/∂uᵢ (5.58). The multipliers "are exactly the local sensitivities of the optimal value with respect to constraint perturbations" — [Convex Optimization, pp. 251-252](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
- [read] Without differentiability, (5.57) gives "a lower bound on the perturbed optimal value, but no upper bound", so the results "are not symmetric with respect to loosening or tightening a constraint". An inactive constraint has a zero multiplier by complementary slackness. The section also gives the shadow-price reading (a firm's profit per unit of resource) — [Convex Optimization, pp. 251-253](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

**CMDP theory**
- [read] Paternain et al. define the perturbation function P(ξ) = max V₀(π) s.t. Vᵢ(π) ≥ cᵢ + ξᵢ. Theorem 1: if rewards are bounded and Slater's condition holds, then P⋆ = D⋆, even though the problem is non-convex in the policy. The proof uses Fenchel-Moreau and shows P(ξ) is concave. The paper uses P(ξ) to prove duality, and a term search found no sensitivity or shadow-price statement — [Constrained RL Has Zero Duality Gap](https://arxiv.org/pdf/1910.13393)
- [read] Theorem 2 of the same paper bounds the gap for a parametrized policy class. The gap is of order the representation error, scaled by (B_r0 + ‖λ⋆‖₁B_r)/(1−γ), where λ⋆ belongs to a perturbed problem. So a large multiplier and γ near 1 both loosen the bound. The paper also says "there need not be a relation" between dual variables and the specifications unless the duality gap is small — [Paternain et al.](https://arxiv.org/pdf/1910.13393)
- [read] Ding, Huan & Ribeiro, Lemma 1: V⋆(ξ) is concave and coordinate-wise non-increasing in the relaxation ξ. Eq. (3): D(λ) is the conjugate of −V⋆(ξ). Lemma 2: for any ξ, λ is a subgradient of −V⋆(ξ) at ξ if and only if λ is a geometric multiplier. They call ∇h(ξ) "the marginal price of relaxing constraint", where h is a designer-chosen relaxation cost. Their equilibrium is ∇h(ξ⋆) ∈ ∂V⋆(ξ⋆) — [Resilient Constrained RL, arXiv](https://arxiv.org/pdf/2312.17194); [AISTATS 2024 record](https://proceedings.mlr.press/v238/ding24a.html)
- [read] Altman's book: §3.3 Theorem 3.6 gives the Lagrangian sup-inf form. Under Slater conditions, optimal policies and multipliers exist and satisfy the Kuhn-Tucker conditions. Chapter 13, "Sensitivity analysis", asks about convergence of values and policies along a sequence of CMDPs, not about derivatives with respect to the limit — [Constrained Markov Decision Processes](https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf)
- [word search] The same extracted book text returned zero hits for "shadow price", "differentiab", "derivative of the value", "perturbation function" and "directional derivative" — [Altman](https://www-sop.inria.fr/members/Eitan.Altman/TEMP/h.pdf)
- [record] Altman & Shwartz 1991 "investigate the sensitivity of the optimal cost and optimal policy to changes in various parameters" through a generic linear program. I did not read the full text, so I do not know whether it treats the constraint limit — [IAOR record](https://iaorifors.com/paper/6019)
- [read] An explicit statement in a survey: "By the envelope theorem (Theorem 36), ∂V∗/∂q_k = λ∗_k". Theorem 36 (attributed to Milgrom & Segal; Danskin) needs a compact action set, jointly continuous f and ∂θf, and a unique maximizer. With several maximizers, V has only one-sided derivatives. The survey links this to the Benveniste-Scheinkman identity — [Rawat 2026, RL for Economics](https://arxiv.org/pdf/2603.08956)
- [read] Recent RL papers state the identity in passing, as known:
  - "λ⋆ = −∂J⋆r/∂d, so λ⋆ is the shadow price of the average-power budget" — [Wang et al. 2026, underwater vehicles](https://arxiv.org/pdf/2606.25680)
  - Proposition 1: ∂J⋆(β)/∂β_k = λ⋆_k "under standard regularity", described as "a standard consequence of the envelope theorem for convex CMDPs and KKT conditions" — [CLAUSE, ICLR 2026](https://arxiv.org/pdf/2509.21035)
- [read] A perturbation analysis of a CMDP's LP with respect to the initial distribution notes that its bounds extend to "sensitivity to constraint threshold" — [Initial Distribution Sensitivity of CMDPs](https://arxiv.org/pdf/2510.00348)

### Inferences
- The identity to quote in the paper is: with bounded rewards, Slater's condition and a fixed initial distribution, V⋆(ξ) is concave and every multiplier is a subgradient of −V⋆(ξ); where V⋆ is differentiable, λ⋆ᵢ = −∂V⋆/∂ξᵢ. This joins Paternain Theorem 1 (concavity, no gap), Ding Lemma 2 (subgradient) and Boyd (5.58) (gradient). No single source I read states it in exactly this form.
- All of these results concern a global value at a fixed initial distribution, under an expected discounted constraint, over a sufficiently rich class of stochastic policies. A critic V(s,b) with a physical rolling budget is a different object. The theorems do not cover it directly (see the question on known problems).
- Since the identity is routinely cited as textbook material in 2026 papers, a theoretical novelty claim is not available. Only the estimator and its empirical validation could be new.

### Gaps
- Altman & Shwartz 1991 full text not read. I do not know whether it covers the limit as a perturbed parameter.
- Chen & Blankenship 2004 was seen only as an abstract [snippet]. Its operators "differ significantly" from standard DP and are shown to be contractions, which may be relevant, but I could not confirm whether it treats the constraint level as a state or discusses multipliers — [record](https://scispace.com/papers/dynamic-programming-equations-for-discounted-constrained-3tklkriegb)
- Piunovskiy & Mao 2000 and Beutler & Ross 1985 were not read. Beutler & Ross is known to me only through BFTQ's citation of it.
- Benveniste & Scheinkman 1979 and Milgrom & Segal 2002 were not read directly. They are known here through the sources above.

---

## Which RL papers read shadow prices or multipliers from a critic conditioned on the budget or threshold?

### Takeaway
None that I found. Budget-conditioned critics exist, and so does the shadow-price statement, but I found no paper that joins them by taking a critic's gradient as the price. The closest work uses the slope for planning or action selection (Boutilier & Lu; RLB) or removes the multiplier through a convex hull without reporting a price (BFTQ).

### Cited Findings

**Closest prior work (must cite)**
- [read] **Boutilier & Lu 2016** is tabular and model-based. Budgeted MDP values are "concave, non-decreasing in budget, and piecewise-linear in the finite horizon case" (Theorem 4). It computes a "bang-per-buck" ratio that expresses the "(per-unit budget) increase", calls these "slopes (BpBs)", and allocates budget to next states in BpB order. This is a marginal value of budget read from a value function. [word search] "Lagrang" returned no hits, so no multiplier comparison was found — [UAI 2016](https://www.auai.org/uai2016/proceedings/papers/246.pdf)
- [read] **Cai et al. 2017 (RLB)** has V(t,b) over remaining auctions and budget, with bids capped by the budget (0≤a≤b). It defines the "value differential D(t,b) = V(t,b+1) − V(t,b)" and fits a neural network NN(t,b) to it, with t and b as inputs. This is a neural model of the marginal value of remaining budget, used to choose bids. The D targets come from exact DP on a small scale rather than from TD learning. [word search] No Lagrange or multiplier comparison found — [Real-Time Bidding by RL](https://arxiv.org/pdf/1701.02490)
- [read] **Carrara et al. 2019 (BFTQ)** is the deep version: the policy and critic take β as input, and the Budgeted Bellman operator uses a convex-hull policy built on a solution set that is "concave and increasing" in β. It compares against FTQ(λ), where "a single training of BFTQ corresponds to 10 trainings of FTQ(λ)", and says of λ calibration "Our purpose is to avoid this calibration phase" — [Budgeted RL in Continuous State Space](https://arxiv.org/pdf/1903.01004)
- [word search] BFTQ uses "multiplier" only for the baseline and "slope" only in the contraction proof. "Shadow" and price language did not appear — [BFTQ](https://arxiv.org/pdf/1903.01004)

**Budget or threshold conditioned, but no price read**
- [read] **Saute RL**: the budget is the initial safety state, z₀ = d, with z_{t+1} = (z_t − l)/γ_l and constraints held "almost surely". This "enables generalization across safety budgets". Its code zeroes the reward when z ≤ 0. [word search] Multipliers appear only for related work and for the baseline — [Saute RL](https://arxiv.org/pdf/2202.06558)
- [read] **CCPO**: critics take the threshold as input, Q̂r(·|εᵢ) and Q̂c(·|εᵢ), yet "the dual variables ηᵢ∗ and λ∗ᵢ are the solutions of" a separate convex problem for each threshold. So a threshold-conditioned critic exists here and the multiplier is still solved for, not read from ∂Q/∂ε. This is a useful contrast for the paper — [CCPO](https://arxiv.org/pdf/2310.03718)
- [read] **Epigraph form (So & Fan, RSS 2023)**: V(x,z), where z is a "cost budget" in units of cost. The authors contrast it with λ, "a ratio representing the cost per unit constraint", which is "harder to interpret". No price is derived in the passages I read — [EFPPO](https://www.roboticsproceedings.org/rss19/p085.pdf)
- [word search] **BCRL (budget-conditioned reachability)**: separate reward and cost critics, no shadow-price or sensitivity language about the budget — [arXiv 2603.22292](https://arxiv.org/pdf/2603.22292)
- [read] **Chow et al. 2015**: yV(x,y) is concave in the CVaR level y, and this justifies linear interpolation. It is a level-conditioned value whose concavity is exploited, but not as a price — [CVaR value iteration](https://arxiv.org/pdf/1506.02188)
- [read] **Budgeted monotonic POMDPs**: proves budget-concavity for that class and fits a random-forest surrogate of value against budget for allocation. It says "prior works implicitly assume and use this budget-concavity" — [arXiv 2408.07192](https://arxiv.org/pdf/2408.07192)

**The reverse direction, or a trained multiplier**
- [read] **State-augmented constrained RL**: puts the Lagrange multiplier in the state and still runs dual dynamics. Saute's appendix also describes it this way. This is λ→policy rather than b→λ — [Calvo-Fullana et al.](https://arxiv.org/pdf/2102.11941)
- [read] **Multiplier models**: learn a model mapping constraint features to multipliers, in ranking and fairness settings. The multiplier is still trained — [Narasimhan et al., NeurIPS 2020](https://papers.nips.cc/paper/2020/file/62db9e3397c76207a687c360e0243317-Paper.pdf)
- [read] **Robot closest to the claim**: Wang et al. 2026 run PPO-Lagrangian on underwater vehicles in the MarineGym simulator with an average-power budget in watts. They call λ⋆ "the marginal task-return cost of one watt, discovered by" dual ascent. [word search] "Slope", "frontier" and "Pareto" have no hits, so there is no critic-read price and no check of the identity — [arXiv 2606.25680](https://arxiv.org/pdf/2606.25680)
- [read] **CLAUSE**: conditions on budgets or prices per query, updates duals by projected ascent, and says the duals "predict the local slope" of the frontier. [word search] "Slope" appears only in that sentence, so I found no test of the claim — [ICLR 2026](https://arxiv.org/pdf/2509.21035)
- [word search] Two legged-robot papers returned nothing relevant for shadow, marginal, conditioned or price terms: [cost-constrained quadruped hardware](https://arxiv.org/pdf/2607.26434), [ECO humanoid](https://arxiv.org/pdf/2602.06445)

**Multi-objective analogue**
- [read] **Envelope MORL**: Q(s,a,ω) is preference-conditioned. Hidden preferences are inferred "through a combination of policy gradient and stochastic search over the preference parameters", with the policy held fixed. This is not a read of ∂V/∂ω as a price. "Envelope" here means the Q-envelope, and "envelope theorem" had no hits — [Yang et al. 2019](https://arxiv.org/pdf/1908.08342)
- [read] **SFOLS**: the optimal value as a function of the weights w forms a "piecewise-linear and convex (PWLC) surface", and corner weights are the points where the surface changes. This is the same kink structure in the dual variable — [Alegre et al. 2022](https://proceedings.mlr.press/v162/alegre22a/alegre22a.pdf)

### Inferences
- BFTQ's hull edge at β has a slope ΔQr/ΔQc, which should be the price at β. So BFTQ implicitly computes the price at every backup and never reports it. A reviewer could raise this, so the paper should address it directly. It also suggests a useful alternative estimator: the hull slope rather than autograd.
- By Ding's eq. (3), a budget-conditioned critic and a λ-conditioned (multi-objective) critic are convex conjugates. Differentiating one in b gives λ, and by Danskin differentiating the other in λ gives the achieved level. This places multi-objective work in the same framework, but it gives the return vector rather than a price.
- A defensible novelty claim is narrow: "first empirical test of autograd-read prices from a single budget-conditioned deep critic against trained multipliers, on a robot with rolling budgets". Claiming a new method without qualification would be exposed to Boutilier & Lu, RLB and BFTQ.

### Gaps
- BFTQ, CCPO, Saute and Ding were read in targeted passages, not end to end. Their supplements and code were not audited, apart from the Saute snippet quoted above.
- Follow-up work citing BFTQ was not traced systematically. A forward-citation check of BFTQ, Boutilier & Lu and Saute in Google Scholar is the most useful remaining step, and I could not do it with these tools.
- Constraint-conditioned actor-critic methods (for example CCAC) and TREBI are remembered as related work but were not checked.

---

## What do related fields do?

### Takeaway
Reading price from the marginal value of remaining capacity is old and central in revenue management, and it is common in bidding and storage. So the idea is not new across fields. What I did not find is its transfer to a deep critic's autograd gradient on rolling budgets, nor a value-read price compared with a trained dual.

### Cited Findings

**Revenue management**
- [read] A bid price is "an opportunity cost", and "Equivalently, bid price can be defined as the marginal value of capacity". The paper notes that the term has "multiple conflicting definitions" — [Revenue Management without Demand Forecasting](https://arxiv.org/pdf/2304.07391)
- [read] That paper estimates proxies "as a function of remaining capacity and time-to-departure", projects them with a neural network, and compares with DP bid prices on revenue and load factor (<1% revenue gap). The comparison is on outcomes, not on price values — [arXiv 2304.07391](https://arxiv.org/pdf/2304.07391)
- [read] Generic bid-price control accepts a request if its revenue is at least the sum of the bid prices of the resources used — [Farias & Van Roy](https://web.mit.edu/vivekf/www/papers/ADP-rm-07-03.pdf). Zhang & Adelman obtain dynamic bid prices through ADP and report that they outperform those from the standard LP — [An ADP approach](https://danzhang.com/papers/network_TS.pdf)
- [read] Akan & Ata construct an epsilon-optimal "bid-price control, where the associated" process "forms a martingale" — [MIT seminar abstract](https://web.mit.edu/orc/www/archive/seminars/2007sp/abs/Ata_abs.html)
- [snippet] The later paper says the optimal shadow price of capacity is a nonnegative martingale, and that non-martingale behaviour may signal a misspecified opportunity cost. The PDF did not extract, so check this before citing — [Stochastic Systems 2015](https://projecteuclid.org/journals/stochastic-systems/volume-5/issue-2/On-bid-price-controls-for-network-revenue-management/10.1214/12-SSY081.full)

**Online allocation and bidding with budgets**
- [read] Dual mirror descent: "they maintain a dual multiplier for each resource that is updated using online mirror descent" — [Balseiro, Lu & Mirrokni](https://arxiv.org/pdf/2011.10124)
- [read] Wu et al. write the optimal bid as v/λ, say "it is extremely difficult to obtain the optimal λ", and use RL to tune λ. This is the trained-parameter counterpart of RLB's value differential, in the same application — [Budget Constrained Bidding](https://arxiv.org/pdf/1802.08365)

**Inventory, storage and resource value**
- [read] Hydro scheduling describes "the derivative of the Bellman value with respect to the storage level" as the system's marginal cost — [Reservoir optimization and ML](https://arxiv.org/pdf/2106.08097)
- [snippet] Actor-critic hydro work uses a critic as "water value function"; I did not verify whether any paper differentiates it — [search lead](https://core.ac.uk/download/358271988.pdf)
- [snippet] Powell's fleet-management ADP uses concave, piecewise-linear value approximations whose parameters are slopes, updated from dual solutions, and reports the marginal value of driver types. This is potentially the closest OR precedent for learning value slopes on resources, but two fetches of the Topaloglu & Powell PDF returned no text and I did not fetch the Simão PDF, so both are unverified — [Topaloglu & Powell](https://typeset.io/pdf/a-distributed-decision-making-structure-for-dynamic-resource-3lrw0nqm79.pdf); [Simão et al.](https://typeset.io/pdf/an-approximate-dynamic-programming-algorithm-for-large-scale-3e3psmn5y0.pdf)

**Economics**
- [read] "Shadow-price learning" treats the state's shadow price as a perceived quantity updated through the first-order and envelope conditions, with value-function learning listed as an alternative. It is an adaptive-learning method, not deep RL — [G. Evans slides, co-authors not confirmed](https://darkwing.uoregon.edu/~gevans/LearnOptTheory7May2018bilbaoGE.pdf)
- [read] Hotelling's rule as defined in an encyclopedia introduction only — [Wikipedia](https://en.wikipedia.org/wiki/Hotelling%27s_rule). Hotelling 1931 was not read, and "Hotelling" had no hits in the Rawat survey.

**Networking**
- [read] Huang & Neely call the attractor role of the dual optimum for the backlog the "counterpart of the 'shadow price' role of Lagrange Multipliers in flow regulation". The backlog is attracted to a dual optimal solution, so a state variable encodes the price — [Delay Reduction via Lagrange Multipliers](https://arxiv.org/pdf/0904.3795)

**Control**
- [read] The envelope or Benveniste-Scheinkman identity gives the derivative of value with respect to a state — [Rawat 2026](https://arxiv.org/pdf/2603.08956)
- [snippet] The adjoint coincides with the value gradient when the PMP conditions are sufficient — [arXiv 2606.23827](https://arxiv.org/pdf/2606.23827)
- [read] Critic derivative training has an older history: "a critic's derivatives are trained to match its target's derivatives" — [Sobolev Training](https://arxiv.org/pdf/1706.04859)

### Inferences
- A virtual queue (Neely) and a rolling budget are mirror images: one tracks accumulated excess and the other remaining allowance. This suggests a simple extra baseline, a price estimated as a function of budget level. This is my analogy, not a result from a source.
- A journal paper should frame this as transferring the bid-price idea to deep RL rather than as a new idea. Reviewers from operations research will recognise it immediately.

### Gaps
- Talluri & van Ryzin 1998 could not be read (HTTP 403 from INFORMS). Its result should be checked directly before citing — [DOI page](https://pubsonline.informs.org/doi/10.1287/mnsc.44.11.1577)
- Token buckets: I found no source that gives a shadow price of bucket depth or refill rate. Unverified leads are [a thesis on token-bucket pricing](https://core.ac.uk/works/2077848) and [Courcoubetis, Kelly & Weber](https://ideas.repec.org/a/inm/oropre/v48y2000i4p535-548.html).
- Optimal power flow, where locational prices are sensitivities, was not searched. It may contain neural-gradient price work.

---

## Has anyone compared a critic-read price with trained multipliers?

### Takeaway
I found no such comparison, so no published agreement figure exists to cite. The nearest numbers are an exact LP check (perturbation slope matches the multiplier) and a tabular Lagrangian run in which the trained multiplier ended 16.7% above the exact price.

### Cited Findings
- [read] **Exact check, no learning**: on a tabular engine-replacement CMDP, the LP multiplier is λ∗ = 6.0909. "Centered budget perturbations reproduce the same slope", so a 0.01 budget increase raises return by 0.0609 — [Rawat 2026, §16.4](https://arxiv.org/pdf/2603.08956)
- [read] **Trained multiplier against exact price**: tabular Lagrangian Q-learning on a carbon-constrained production CMDP, ten seeds. λ settles near 1.40 against the LP's λ∗ = 1.20: "The final λ is 16.7% above λ∗". Emissions are 28.44 ± 3.60 against a budget of 31.35, and return is 178.4 ± 4.0 against 186.4. The LP randomizes at one state while Q-learning recovers a deterministic approximation — [Rawat 2026](https://arxiv.org/pdf/2603.08956)
- [read] **Deep RL frontier slope, described qualitatively**: Spoor et al. train 25 fixed-λ policies and say "the local slope of the curve, dR/dC, captures the sensitivity of return to changes in the allowed cost". [word search] I found no numerical comparison of that slope with λ. They define λ∗ as where the cost curve crosses the limit — [Towards a Practical Understanding of Lagrangian Methods, arXiv 2510.17564](https://arxiv.org/pdf/2510.17564)
- [read] Fixed λ∗ is not exact even so: on SafetyPointCircle1-v0 at limit 25.0 the achieved cost was 4.75 ± 3.93. Preliminary SAC-Lag results (5 seeds) have a different constraint geometry from PPO-Lag — [Spoor et al.](https://arxiv.org/pdf/2510.17564)
- [read] Stooke et al. report that "the cost and multiplier values oscillated throughout training" in deep RL — [PID Lagrangian](http://proceedings.mlr.press/v119/stooke20a/stooke20a.pdf)
- [read] Wang et al. report λ trajectories that "rise while the policy is over budget and then settle", and call the result "a working multiplier" — [arXiv 2606.25680](https://arxiv.org/pdf/2606.25680)

### Inferences
- The 20% target has no published basis, and a trained SAC-Lagrangian multiplier is a weak reference. In the only comparable number I found, the trained multiplier was 16.7% off the exact price in a small tabular problem, so 20% agreement with SAC-Lagrangian could occur even if both are wrong, or fail even if the critic is right.
- A stronger design has three references: (a) an exact LP or DP answer on a small CMDP, (b) a finite-difference slope from fixed-budget retrains at b ± δ, and (c) SAC-Lagrangian multipliers. Spoor's fixed-λ sweep and Rawat's centered perturbations are precedents for (b).
- Agreement should be pre-registered per limit, with seed spread for both estimators, since trained multipliers oscillate. The project's working rules also apply here: every seed counts, and no big claims before 5 seeds.
- Include a limit that does not bind. Complementary slackness predicts a price near zero, which is a sharper test than a ratio.

### Gaps
- Spoor's supplementary figures were not inspected visually, so a slope-against-λ reading from the plots remains possible.
- A paper that performs this comparison under different vocabulary may exist (see the search log).

---

## What could make dV/db unreliable?

### Takeaway
There are six concrete risks, and the first two are the most serious for this plan. The concavity theorems are for expected-budget problems, not physical rolling budgets, and a state-conditioned gradient is not the same object as a global scalar λ. Kinks, critic gradient error, discount and unit mismatch, and SAC's entropy term are also documented or directly implied.

### Cited Findings

**1. The concavity theorems assume in-expectation budgets**
- [read] Boutilier & Lu: "we will require only that policies meet this constraint in expectation" — [UAI 2016](https://www.auai.org/uai2016/proceedings/papers/246.pdf)
- [read] BFTQ: the policy outputs the next budget, with dynamics P(s′|s,a)δ(β′−βₐ), and "the budget β is neither fixed nor constant... but instead evolves as part of the dynamics". The objective is E[Gc] ≤ β — [BFTQ](https://arxiv.org/pdf/1903.01004)
- [read] Saute's budget is instead a fixed update, z_{t+1} = (z_t − l)/γ_l, under an almost-sure constraint — [Saute RL](https://arxiv.org/pdf/2202.06558)

**2. Non-concave value**
- [read] With deterministic policies, the value is "piecewise constant and monotone" in budget, and stochastic policies give the concave hull — [Boutilier & Lu](https://www.auai.org/uai2016/proceedings/papers/246.pdf)
- [read] Size of the difference on one example: best deterministic return 3.4545 against randomized 4.6727 at the same budget — [Rawat 2026, §16.4](https://arxiv.org/pdf/2603.08956)
- [read] With a hard budget and neural approximation: "D_t(b) fluctuates heavily when b is very small, and later keeps decreasing to 0" — [RLB](https://arxiv.org/pdf/1701.02490)
- [read] On a learned deep RL frontier: on SafetyHopperVelocity-v1 the slope is steep below cost ~25, flatter from 25 to 300, then restrictive again above 300, which they call "non-monotonic trade-off behavior". The authors note in their appendix that seed variance may contribute to some irregular curves — [Spoor et al.](https://arxiv.org/pdf/2510.17564)

**3. Kinks where a limit stops binding**
- [read] One-sided behaviour at non-differentiable points — [Boyd §5.6](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf); [Rawat Theorem 36](https://arxiv.org/pdf/2603.08956); subgradient set — [Ding Lemma 2](https://arxiv.org/pdf/2312.17194)
- [read] Slack gives λ⋆ = 0. An infeasible budget pushes the trained multiplier to its ceiling, where it "saturates" — [Wang et al., Remarks 1 and 3](https://arxiv.org/pdf/2606.25680). Infeasible perturbations yield an unbounded multiplier in theory — [Paternain et al.](https://arxiv.org/pdf/1910.13393)
- [read] A single-constraint CMDP optimum is a randomized mixture of two deterministic policies (BFTQ citing Beutler & Ross 1985, Theorem 4.4) — [BFTQ appendix](https://arxiv.org/pdf/1903.01004)
- [snippet] In LP, dual values may not equal shadow prices at a degenerate optimum, and two-sided prices have been proposed. These are from records only — [Aucamp & Steinberg 1982](https://link.springer.com/content/pdf/10.1057%2Fjors.1982.118.pdf); [Gal 1986 record](https://portal.mardi4nfdi.de/wiki/Publication:1075251); [Ho 2000](https://informatica.vu.lt/journal/INFORMATICA/article/314)

**4. Critic approximation error**
- [read] RLB chose to model the differential instead of V because V is much larger than the differences: "if we take V(t,b) as our target to approximate, it is difficult to give a low deviation", and "we care about the value of V(t−1,b−δ)−V(t−1,b)". This is the closest direct evidence for this plan — [RLB](https://arxiv.org/pdf/1701.02490)
- [read] For action gradients: "Learning the critic by value rather than by action-gradient of the value relies on hazy smoothness assumptions", and "the critic learns action-value gradients implicitly". Proposition 3.1 points to the gradient of the evaluation error as what matters — [MAGE](https://arxiv.org/pdf/2004.14309)
- [read] BFTQ's operator fails to be a contraction in a counter-example "in which the slope of Q can be arbitrary large", and BFTQ's sketch restricts to L-Lipschitz frontiers with L < 1/γ − 1. Steep prices are therefore the theoretically hard case — [BFTQ](https://arxiv.org/pdf/1903.01004)

**5. Discount and unit mismatch**
- [read] The most direct statement I found: "the cost critic bootstraps the budget-centered power with discount γ", "whereas the constraint is enforced on the undiscounted episode-average power", and "the two averages agree as γ→1 and are close at γ=0.99" — [Wang et al., Remark 2](https://arxiv.org/pdf/2606.25680)
- [read] Units vary between formulations: Saute divides by γ_l each step; Ding rescales the threshold by (1−γ) — [Saute](https://arxiv.org/pdf/2202.06558), [Ding](https://arxiv.org/pdf/2312.17194). Rawat's budget is a "normalized discounted budget" — [Rawat §16.4](https://arxiv.org/pdf/2603.08956)

**6. SAC's entropy term**
- [read] SAC-Lag uses the maximum-entropy actor objective — [Spoor et al. supplement](https://arxiv.org/pdf/2510.17564). A robot SAC-Lagrangian has two multipliers, α for entropy and λ for safety — [Ha et al. 2020](https://arxiv.org/pdf/2002.08550)

### Inferences
These are my own derivations and readings. No source states them, so they need independent checking before use.
- **Risk 1 is the key theoretical gap.** In Boutilier and BFTQ the policy allocates next-state budgets so that spending holds in expectation, and concavity comes from that freedom. A token bucket fixes b′ = b − c + f (f is the refill), like Saute. So V(s,b) need not be concave, and the hull result does not apply. RLB's fluctuating D at small b is consistent with this. Hard and expected limits should approach each other only when the window is long relative to mixing time. The paper therefore needs either a proof for its own setting or an empirical concavity check.
- **State-dependent price against a scalar λ.** Write R for reward and f for refill. Applying the envelope theorem to V(s,b) = max_a {R + γE[V(s′, b−c+f)]} gives ∂V/∂b(s,b) = γE[∂V/∂b(s′,b′)], where V is differentiable, the budget update is uncapped, and the budget does not restrict the action set.
  - The current-value price therefore grows by 1/γ per step along optimal paths when reward is discounted and the budget is not. This has the form of a discrete Hotelling rule, and it is the concrete form of the discount mismatch.
  - With Saute's update and γ_l = γ, the price is instead a martingale, the same form as Akan & Ata's result, although their setting is continuous-time revenue management. With γ_l ≠ γ it drifts by γ/γ_l per step.
  - Averaging over the replay buffer would therefore be wrong if a drift exists. The fair comparison is E over initial states of ∂V(s₀,b₀)/∂b₀.
- **Token-bucket cap.** With b′ = min(B, b−c+f), the local price is zero whenever the cap binds, so a kink sits at the cap.
- **Level and rate prices differ.** The same recursion gives ∂V/∂f = Σ_k γᵏE[∂V/∂b_k], which equals T·(∂V/∂b) over T remaining refills if the budget binds throughout. The two prices are therefore not interchangeable, and the ratio depends on the remaining horizon, not on a constant.
- **A cheap diagnostic follows.** Test the residual ∂V/∂b(s,b) − γE[∂V/∂b(s′,b′)] on the trained critic. A large residual flags an unreliable price without any multiplier run. The same recursion could serve as a Sobolev-style derivative target or regularizer. This extends Akan & Ata's monitoring suggestion, which I saw only as a snippet.
- **Autograd against finite differences.** Autograd reads a local slope, which is the most exposed to ReLU kinks and fit noise. A centered difference across a band matches the theory better (Rawat's method) and gives a cheap robustness check. Input-concave architectures could enforce concavity, but they would also hide violations, so they should not be used to validate the claim.
- **Saute boundary.** Saute's reward zeroing at z ≤ 0 builds a steep edge into V. Gradients near empty budgets may reflect the shaping rather than the task.
- **Soft values.** A soft critic includes entropy, so dV/db is the price in soft return. SAC-Lagrangian's λ also acts on a soft objective, so they may be comparable, but only if the budget-conditioned critic's α matches that of the Lagrangian run. Differing α values would add a scale difference unrelated to the method.
- **MAGE concerns action gradients.** Applying it to the budget input is my extension. No source addresses budget-input gradients.

### Gaps
- I found no source quantifying gradient error of a TD-trained critic with respect to a non-action input.
- I found no CMDP result on concavity of V(s,b) for a physical rolling budget.
- A sliding-window bandit paper was summarised by the search tool as saying generic state augmentation can have exponential dependence on window length, but my term search of the PDF did not confirm this. This matters if the rolling window is exact rather than a token bucket — [arXiv 2610.08745](https://arxiv.org/pdf/2610.08745)
- I could not confirm a standard discount conversion for SAC-Lagrangian limits. The WCSAC PDF returned no usable text, and the lines I searched in [FSRL's policy code](https://github.com/liuzuxin/FSRL) showed no conversion.

---

## What was searched, and what found nothing?

### Takeaway
Direct queries for this method returned nothing matching, across several wordings. This is weak evidence of absence, because the search tool returns only a few results per query and summarizes them, and it does not reach citation graphs, Google Scholar or paywalled text.

### Cited Findings
**Web queries with no paper matching the target (all run on 2026-11-04)**
- "shadow price from gradient of budget-conditioned critic reinforcement learning"
- "critic gradient with respect to budget input dual variable without training Lagrangian multiplier robot"
- "marginal value of remaining budget from Q-network budget in state deep RL dual variable recovered"
- "envelope theorem Lagrange multiplier autograd neural value network budget input constrained reinforcement learning without dual ascent"
- "value function sensitivity constraint threshold learned multiplier compared numerically 'shadow price' constrained reinforcement learning estimate finite difference"
- "rolling window cost budget constrained reinforcement learning robot thermal energy budget in observation" (no thermal or rolling-window robot work with a budget input)
- "multi-robot resource allocation shadow price learned value function gradient reinforcement learning battery energy budget"
- "bid price from value function gradient remaining capacity deep reinforcement learning network revenue management" (no deep RL gradient bid price)
- "neural network value function network revenue management bid prices approximate dynamic programming deep learning marginal capacity" (ADP work and a data-driven NN paper only)
- "Hotelling shadow price deep reinforcement learning resource stock marginal value critic"
- "social cost of carbon deep reinforcement learning value function gradient shadow price neural network automatic differentiation"
- "water value marginal value of storage reinforcement learning critic gradient state of charge" (critics exist, no gradient read found)
- "token bucket reinforcement learning constraint dual price virtual queue Lagrange multiplier estimate"
- "token bucket parameters shadow price effective bandwidth Courcoubetis Kelly Weber buffer capacity prices"

**Queries that did find what they targeted**
- "Lagrange multiplier equals derivative of value with respect to constraint threshold constrained MDP deep RL" (papers on the setup; the statement came from reading full texts)
- "Resilient constrained reinforcement learning perturbation function derivative dual multiplier constraint relaxation"
- "Altman Shwartz Sensitivity of constrained Markov decision processes Annals of Operations Research 1991"
- "Chen Blankenship dynamic programming equations discounted constrained stochastic control..." (paper found; no multiplier detail)
- "Boutilier Lu budget allocation weakly coupled constrained MDPs value concave piecewise linear in budget"
- "Akan Ata bid-price controls network revenue management martingale characterization optimal bid prices"
- "shadow prices linear programming degeneracy left right shadow price kink sensitivity analysis just be careful" (survey and related records found; the "Just be careful!" paper itself was not retrieved)
- "Sobolev training neural network value function gradient accuracy critic derivative error costate deep learning HJB"
- "Powell approximate dynamic programming concave value function slopes resource marginal value dual fleet management" (snippets only)
- "Benveniste Scheinkman differentiability of the value function envelope theorem dynamic programming interiority Rincon-Zapatero Santos" (records only: [Rincón-Zapatero & Santos 2009](https://econpapers.repec.org/RePEc:eee:jetheo:v:144:y:2009:i:5:p:1948-1964); [Clarke differentials paper](https://arxiv.org/pdf/2509.17103))
- "Huang Neely delay reduction via Lagrange multipliers stochastic network optimization queue backlog multiplier"

**Full-text term searches with no price or sensitivity hit on the budget** (terms such as shadow, sensitiv, marginal, slope, derivative): BFTQ, Saute, CCPO, BCRL, Wang et al. (beyond Remark 1), the two legged-robot papers above.

### Inferences
- Wording is the main risk. A paper might say "marginal utility of budget", "dual recovered from value", "value of a unit of constraint" or "budget sensitivity". I tried several variants, not all.
- Most results came from arXiv and open proceedings. Operations research journals are under-covered because of paywalls, and that is exactly where related work is most likely.

### Gaps
- Not done: forward-citation search on BFTQ, Boutilier & Lu, Saute and Paternain; Google Scholar exact-phrase searches; a look at optimal power flow pricing with neural surrogates. These are the recommended next checks before making a novelty claim in a journal submission.
- Fetch failures: Talluri & van Ryzin (403), Akan & Ata 2015 PDF (no text), Powell PDFs (no text), WCSAC PDF (no text), Hotelling lecture notes (no text).
- WebFetch returned paraphrases for some arXiv abstract pages, so I used direct PDF text extraction for all quoted statements.
