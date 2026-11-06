# Rolling and threshold-conditioned constraints in RL: prior-work novelty check

Searches ran on 2026-11-04. The claim tested: rolling token-bucket budgets, b(t+1) = min(B, b(t) + r - c(t)), with cost over any T-step window at most r*T + B, are given to the policy as part of the goal, and one training run covers many limits through hindsight relabeling.

Evidence tags used on every finding:
- [T] I read the paper's own text directly (PDF converted to text).
- [H] Paper HTML or PDF read through the fetch tool's summarizer, which returned quotes and paraphrase. Re-read before quoting in a journal paper.
- [A] Abstract only.
- [S] Search-result excerpt only. Treat as a lead.

## Does any earlier work condition a policy on constraint limits AND use rolling budgets?

### Takeaway
I did not find a paper with all three parts: a capped, replenishing budget with an any-window bound, the limits (r, B) as a policy input, and relabeling across limits. No single element is new on its own, though, and two papers are one step from the combination: Qiu et al. 2022 (same recursion, DQN, updates all bucket levels from one trace, but one (r, b) per network) and SeqRoute 2026 (hindsight budget relabeling by name, but a per-session total). This is "not found", not proof of absence; the citation trail is the weakest part of the search.

### Cited Findings

Element-by-element map of the closest papers:

| Paper | Rolling, capped budget? | What the policy receives | Where data for other limits comes from | Evidence |
|---|---|---|---|---|
| [Qiu et al. 2022, TCAD](https://arxiv.org/pdf/2208.00485) | Yes, n[t+1] = min(b, n[t] - a[t] + r) | Token level, through one Q-output per token state; r and b are not inputs | All token levels updated from one trace, for one (r, b) | [T] |
| [Chakrabarti et al. 2020](https://arxiv.org/html/2010.13737v2) | Yes, same recursion | Token count and input; not RL (value iteration) | Thresholds computed for specified bucket parameters | [H], partly [T] |
| [Zakeri et al. 2025](https://arxiv.org/pdf/2511.01406) | Reflected queue with no cap | Virtual queue and age of information; limit not in the state | Not addressed | [T] |
| [Sharma et al.](https://ar5iv.labs.arxiv.org/html/1807.08315) | Yes, a battery: e' = min(e - e_TX + e_H, N_e) | Battery level; capacity is not an input | Not addressed | [H] |
| [Cheema et al. 2023](https://arxiv.org/pdf/2310.08583) | Recovering fatigue state, not a cost budget | Fatigued motor units MF; F, R, r not listed in the observation space | MF randomized at reset; one (F, R, r) trained, several tested | [T] |
| [SDGD (Chen et al. 2026)](https://arxiv.org/pdf/2605.02777) | No refill, but sub-trajectory limits l_k | Cost limit l, conditioned in a diffusion planner | Each trajectory's own cumulative cost builds the conditioning | [T] |
| [SeqRoute (Xu et al. 2026)](https://arxiv.org/pdf/2605.25424) | No, per-session total | Remaining budget | Replay through the budget transition for five virtual budgets | [T] |
| [Brahmanage and Kumar 2026](https://arxiv.org/pdf/2603.22292) | No, depleting | (state, budget) | Budget sampled onto each offline transition | [T] for the sampling, [H] for the rest |
| [CDT (ICML 2023)](https://arxiv.org/pdf/2302.07351) | No, cost-to-go | Target cost and reward return tokens | Relabels infeasible target pairs onto safe Pareto trajectories | [T] |
| [CCPO](https://arxiv.org/pdf/2310.03718) | No, discounted total | Fixed threshold epsilon | N behavior policies plus value generalization | [T] |
| [Sauté](https://arxiv.org/pdf/2202.06558) | No, only depletes | Remaining budget z | Initial budget resampled per episode | [T] |
| [Linear bandits, sliding window (2026)](https://arxiv.org/pdf/2610.08745) | Exact sliding-window feasibility | Recent action history; not a budget or limit | Constraint set treated as fixed | [H], partly [T] |
| [Sliding-window auto-bidding (2026)](https://arxiv.org/pdf/2607.25233) | W-episode window on a ratio | Per-episode return and cost tokens; window not visible to the controller | Ground-truth tokens from logs; no relabeling described | [H], partly [T] |

The papers that come closest to breaking the claim:

- **Qiu et al.** use the identical recursion with a DQN. They state that "token states are a deterministic function of offloading actions and our inputs (and metrics) are statistically independent of actions" — [Qiu et al.](https://arxiv.org/pdf/2208.00485) [T]
- **Qiu et al. reuse data across bucket levels.** They compute and update "Q-values for all token bucket states n̄ at the same time without resampling training data based on policy, i.e., avoid doing proper reinforcement learning" — [Qiu et al.](https://arxiv.org/pdf/2208.00485) [T]
- **Qiu et al. do not cover many limits with one network.** The DQN produces 2M-P-N+2 outputs, and training data is generated "Given a token bucket configuration". Their figures vary r over 0.05, 0.1 and 0.25 and b from 1 to 20 — [Qiu et al.](https://arxiv.org/pdf/2208.00485) [T]
- **SeqRoute uses the term and the mechanism.** "HBR retroactively asks: what would have happened if this session operated under a tighter budget?" It replays the same sequence "through the deterministic budget transition (Eq. 3)" and the final dataset is "the union over all B0 values" — [SeqRoute](https://arxiv.org/pdf/2605.25424) [T]
- **SeqRoute's budget is a total.** The constraint is on a global session budget B_total, with b_{t+1} = b_t - C/B_total and a 4-turn horizon — [SeqRoute](https://arxiv.org/html/2605.25424) [H]; the sum against B_total is also in the [PDF](https://arxiv.org/pdf/2605.25424) [T]
- **Brahmanage and Kumar relabel at transition level.** The budget is drawn from U[Q*_C(s,a), δ_max], the augmented data is sampled dynamically, and the paper says this "acts as data augmentation rather than filtering" — [Brahmanage and Kumar](https://arxiv.org/pdf/2603.22292) [T]
- **SDGD is the nearest conditioned paper with sub-episode cost limits.** Its objective is E[C_k(τ)] ≤ l_k for k in 0..K, where C_k is "the cost of sub-trajectories with predefined horizon T_k", described as covering "time-varying limits (K > 0) through segmented cost limits" — [SDGD](https://arxiv.org/pdf/2605.02777) [T]
- **SDGD uses one model for many limits.** "All results use a single trained SDGD model; no retraining is performed when the cost limit changes." It tests a limit changing from 1 to 3 to 10 within an episode. Conditioning is built from each trajectory's cumulative cost, so labels come from achieved cost — [SDGD](https://arxiv.org/pdf/2605.02777) [T]
- **SDGD lacks the token-bucket parts.** In the passages I read there is no refill rate, no burst cap, no budget state and no any-window bound. Its "relabeling" changes the reward predictor's target and leaves "the cost limit conditioning unchanged" — [SDGD](https://arxiv.org/pdf/2605.02777) [T]
- **The bound itself is classical.** Network calculus defines an arrival curve by R(t) - R(s) ≤ α(t - s) for all s ≤ t, noting "the condition is over a set of overlapping intervals", and uses γ_{r,b}(t) = rt + b because of its relationship with leaky buckets — [Le Boudec and Thiran](https://leboudec.github.io/netcal/latex/netCalBook.pdf) [T]
- **Sauté hints at reuse.** Its limitations section suggests countering lost efficiency "in the setting where the safety cost function is given, by exploiting the known safety state transitions" — [Sauté](https://arxiv.org/pdf/2202.06558) [T]

### Inferences
- **Status of the claim:** the three-part combination survives this search. Claims of "first token bucket in RL", "first budget relabeling" or "first any-window guarantee" would not survive. A defensible wording is limited to the combination, and it should cite Qiu 2022, SeqRoute 2026, Brahmanage and Kumar 2026, SDGD 2026 and network calculus.
- **Expected reviewer objection:** "Add (r, b) to Qiu's input" or "swap SeqRoute's recursion for a token bucket". The paper needs an argument for why the combination is not trivial. Two candidates follow.
- **Difference from Qiu:** their full-level update relies on inputs being independent of actions, which they themselves call avoiding proper RL. On a controlled robot, actions change the next state, so relabeling is genuine off-policy learning and carries its bias. That assumption does not carry over.
- **Difference from SeqRoute:** a depleting total has one crossing point per budget. With a cap and refill, the same cost trace gives a different budget history for each (r, B), including overflow losses. This is my reasoning, not a statement in any paper.
- **Same recursion as a virtual queue.** Set Q = B - b. Then b(t+1) = min(B, b + r - c) becomes Q(t+1) = max(0, Q + c - r), which is [Zakeri et al.](https://arxiv.org/pdf/2511.01406) Eq. 4 with α = c and α_max = r. Their queue has no cap and only targets a long-run average. A reviewer from networking will likely notice this.
- **The bound needs enforcement.** Telescoping b(t+1) ≤ b(t) + r - c(t) over a window gives Σc ≤ rT + b(s) - b(s+T) ≤ rT + B only if b(s+T) ≥ 0. Qiu enforces this as a hard rule (offloading "needs n[t] ≥ 1"). A learned policy with a soft penalty has only a conditional bound, and the paper should say so.
- **The cap is what makes it rolling.** [McMahan and Zhu](https://proceedings.mlr.press/v238/mcmahan24a/mcmahan24a.pdf) allow "refueling" through negative costs, but with c' = c - r their constraint bounds only sums from the start of the episode. Banked credit is unlimited, so no any-window bound follows.

### Gaps
- No search can prove absence. Paywalled networking venues and non-English work were reachable only through abstracts at best.
- Whether Qiu trained a separate DQN per (r, b) is not stated. I infer it from the output size depending on M, P and N.
- SDGD's windows could be read as overlapping if the horizons T_k overlap. The paper calls them "segmented" and gives no any-window statement, but its full appendix was not read closely.

## Which RL papers use token buckets, leaky buckets, sliding windows, moving averages, rate limits, peak or short-horizon cost constraints? As input or goal? Relabeled?

### Takeaway
Token-bucket state in RL exists in edge offloading, and equivalent recursions exist in energy harvesting and Lyapunov DRL. In all of these the limit parameters are fixed design values, not policy inputs. Window constraints exist in bandits and auto-bidding, without limit conditioning or relabeling. Peak constraints are per-step or per-state. I found no moving-average cost constraint paper.

### Cited Findings

Token and leaky buckets:
- Chakrabarti et al. use n[t+1] = min(b, n[t] - a[t] + r) with value iteration, and compute "thresholds for a specified error measure and token bucket parameters" — [arXiv 2010.13737](https://arxiv.org/html/2010.13737v2) [H]; [SEC 2021 PDF](https://www.cse.wustl.edu/~lu/papers/sec21.pdf) [T]
- Qiu et al. is the deep RL version (see the table above) — [arXiv 2208.00485](https://arxiv.org/pdf/2208.00485) [T]
- A follow-up on object detection is by the same group. Its abstract does not mention a token bucket or RL; the authors' slides do, but I saw those only as an excerpt — [arXiv 2410.18919](https://arxiv.org/abs/2410.18919) [A]
- RACER derives a per-task QoS constraint from a token bucket; excerpts say the learned policy sets the shaping, which would make the bucket an action, not a state. Full text could not be fetched — [IEEE Access 2020](https://doi.org/10.1109/access.2020.3033421) [A]
- An IJCNN 2024 paper uses a token bucket to regulate offloading from edge devices — [DOI](https://doi.org/10.1109/ijcnn60899.2024.10651213) [A]
- A book chapter applies RL to enhance a leaky-bucket usage parameter control mechanism — [DOI](https://doi.org/10.1201/9781315220178-8) [A]
- In adaptive rate limiting for microservices, RL adjusts the limit: the state includes the current threshold and the action is a percentage change. Token bucket is only a baseline — [arXiv 2511.03279](https://arxiv.org/pdf/2511.03279) [T]

Equivalent recursions under other names:
- A DQN state contains the virtual queue, with Q(t+1) = max[Q(t) + α(t) - α_max, 0]; the constraint is a long-run average and results span several α_max values — [Zakeri et al.](https://arxiv.org/pdf/2511.01406) [T]
- Energy-harvesting RL uses e_{n+1} = min(e_n - e_TX(a_n) + e_Hn, N_e), with battery in the state (b, e, h) — [Sharma et al.](https://ar5iv.labs.arxiv.org/html/1807.08315) [H]
- State-augmented constrained RL puts Lagrange multipliers in the state and samples (s, λ) in training, with "no dual dynamics" used during training. Multipliers update per epoch, constraints are long-run averages, and thresholds are not inputs — [Calvo-Fullana et al.](https://ar5iv.labs.arxiv.org/html/2102.11941) [H]

Sliding windows:
- Every realized window must satisfy the constraint exactly: (x_{t-w+1},…,x_t) ∈ C. One example is an aggregate budget, with w = 5 and B = 0.40 in LLM routing. Feasibility is defined from the action history. No budget input or relabeling was found — [arXiv 2610.08745](https://arxiv.org/html/2610.08745) [H]
- Auto-bidding bounds an efficiency ratio over every W-episode window, but "The information boundary of the DT is confined to" a single episode, so the controller does not see the window state — [arXiv 2607.25233](https://arxiv.org/pdf/2607.25233) [T]. The experiments use W = 7 — [HTML version](https://arxiv.org/html/2607.25233) [H]
- A 2025 UAV paper uses "dynamic sliding-window assessments" with Lagrangian multipliers. This looks like a dual-update detail, not a policy input — [DOI](https://doi.org/10.1109/icct67417.2025.11374021) [A]
- Formal methods have window mean-payoff objectives on MDPs, computed over "a finite window that slides over the entire path". These are synthesis results, not RL — [Bordais et al.](https://arxiv.org/pdf/1812.09298) [T]

Peak and short-horizon:
- Peak-cost RL constrains max c(s) ≤ b over reachable states, per state rather than per window — [arXiv 2607.15457](https://arxiv.org/pdf/2607.15457) [T]
- Earlier work studies MDPs with peak constraints for discounted and average rewards — [arXiv 1901.07839](https://arxiv.org/pdf/1901.07839) [T, abstract and keywords]

False positives that look windowed but are not:
- The "14 steps" in constrained MARL is the imagination horizon; the limit is on expected episode cost — [arXiv 2610.21319](https://arxiv.org/html/2610.21319) [H]
- AnySearch's "sliding window of 20" tracks accuracy for choosing training budgets — [arXiv 2609.00813](https://arxiv.org/html/2609.00813) [H]
- TREBI's "real-time budget" means a budget picked at deployment, with C(τ) ≤ b at trajectory level — [arXiv 2306.00603](https://arxiv.org/pdf/2306.00603) [T]
- EMA in CSPO smooths a constraint-related weight, not a cost constraint — [arXiv 2606.14415](https://arxiv.org/html/2606.14415) [S]

### Inferences
- None of the rolling-budget RL papers I read treats (r, B) as a goal. They fix them as system parameters, because in networking the operator sets the limit once.
- Relabeling appears only in the per-episode family and, as a full-level update under one fixed configuration, in Qiu.
- Motor heat behaves like a leaky budget, so robotics thermal work is a physical analogue. A [2026 quadruped paper](https://arxiv.org/abs/2605.27046) [S] adds a thermal model and residual policy. I did not confirm its inputs.

### Gaps
- RACER, the IJCNN 2024 paper and the UAV paper were not read in full.
- Moving-average or exponentially forgetting cost constraints: no paper found, in three differently worded searches.
- I did not confirm whether the auto-bidding planner receives ρ_tgt as a model input; the fetched reading says it enters through a scoring penalty.

## CCPO, Sauté, Simmer, Budgeted RL and follow-ups: total or rolling? Policy conditioned? Where does data for other limits come from?

### Takeaway
All four named papers use a per-episode discounted or almost-sure total, and none replenishes it. All condition the policy on the budget in some form. None relabels: other limits come from behavior policies plus value generalization (CCPO), resampled initial budgets (Sauté, Budgeted RL), or a training schedule (Simmer).

### Cited Findings

CCPO (Yao et al., NeurIPS 2023):
- Budget: V_c^π(μ0) = E[Σ_{t=0}^∞ γ^t c_t] ≤ ε, Eq. 1; evaluation uses CV = max{0, Σ_τ c_t - ε}. A discounted total — [CCPO](https://arxiv.org/pdf/2310.03718) [T]
- Conditioning: π(·|ε): S × A × E → [0,1], with ε fixed rather than tracked as a depleting state — [CCPO](https://arxiv.org/pdf/2310.03718) [T]
- Data: "collected through a limited set of thresholds", from N behavior policies. Experiments collect at {20, 40, 60} and evaluate at {10, 15, ..., 70} — [CCPO](https://arxiv.org/pdf/2310.03718) [T]
- Other limits: Q(s,a|ε) = ψ(s,a)^T z(ε) generalizes the critics, then the M-step fits the policy by supervised learning across sampled thresholds — [CCPO](https://arxiv.org/pdf/2310.03718) [T]
- A word search of the arXiv v2 text finds no "relabel", "hindsight", "replay" or "buffer" — [CCPO](https://arxiv.org/pdf/2310.03718) [T]
- Its priority statement is limited: "the first successful online safe RL approach capable of achieving zero-shot adaptation for unseen thresholds while preserving safety" — [CCPO](https://arxiv.org/pdf/2310.03718) [T]

Sauté (Sootla et al., ICML 2022):
- Budget: J = d - Σ γ_l^t l(s_t, a_t), with nonnegative costs and z_{t+1} = (z_t - l(s_t,a_t))/γ_l, z_0 = d. "the available safety budget diminishes over time" — [Sauté](https://arxiv.org/pdf/2202.06558) [T]
- Conditioning: z is part of the augmented state, and d "enters the problem formulation as the initial value of the safety state" — [Sauté](https://arxiv.org/pdf/2202.06558) [T]
- Data: "we can randomly sample different safety budgets at the beginning of the episode". The zero-shot experiment samples uniformly from [5, 100] — [Sauté](https://arxiv.org/pdf/2202.06558) [T]

Simmer (Sootla et al., NeurIPS 2022; arXiv title "Effects of Safety State Augmentation on Safe Exploration"):
- Budget: same recursion, z_0 = d, with z_t as "the remaining safety budget"; the policy is π(·|s, z) — [Simmer](https://ar5iv.labs.arxiv.org/html/2206.02675) [H]
- The controller changes the training budget between epochs, moving d_k from d_start to d_target. The fetched text describes a PI controller (with optional low-pass filter and anti-windup) and a Q-learning variant, not a PID as the request phrased it — [Simmer](https://ar5iv.labs.arxiv.org/html/2206.02675) [H]
- Data: on-policy data is emptied each epoch and off-policy buffers may grow. No relabeling is described, and sharing across budgets is not addressed — [Simmer](https://ar5iv.labs.arxiv.org/html/2206.02675) [H]

Budgeted RL (Carrara et al., arXiv 1903.01004):
- Budget: E[G_c | s0] ≤ β with G_c = Σ γ^t R_c, and the policy "takes β as input" — [BFTQ](https://arxiv.org/pdf/1903.01004) [T]
- Dynamics: the policy also outputs the next budget, δ(β' - β_a), so "the budget β is neither fixed nor constant". There is no exogenous refill — [BFTQ](https://arxiv.org/pdf/1903.01004) [T]
- Data: exploration samples "initial budget β ∼ U(B)" and stores (s, β, a, β_a, R, C, s'). Function approximation is used to "generalise to nearby states and budgets". I found no relabeling or resampling of stored budgets — [BFTQ](https://arxiv.org/pdf/1903.01004) [T]

Follow-ups and relatives, all per-episode:
- UCPO: Z_{t+1} = Z_t + R_{t+1}, limits are "defined over the undiscounted cumulative cost of an episode", z0 sampled from [-30, 0], and "we do not retrain the agent" for other budgets — [arXiv 2606.14029](https://arxiv.org/html/2606.14029) [H]
- Safe In-Context RL: policies are conditioned on cost-to-go, with "G_c,0 = δ" at episode start, for "regulating episode-level costs" — [arXiv 2509.25582](https://arxiv.org/pdf/2509.25582) [T]
- CDT: C_t = Σ_{t'=t}^T c_t', with "Pareto frontier-oriented data augmentation by target return relabeling"; κ is drawn from U(c_min, c_max) and infeasible targets map to a safe Pareto trajectory — [arXiv 2302.07351](https://arxiv.org/pdf/2302.07351) [T]
- GAS: cost target resampled between the segment's true cost and the dataset maximum; the relabeled values feed the goal functions, while losses use true values — [arXiv 2602.05323](https://arxiv.org/html/2602.05323) [H]
- CCAC (ICLR 2025): adapts to varying thresholds, and critics are constraint-conditioned. Body not read because the PDF would not convert and OpenReview returned a verification page — [NSF PAR record](https://par.nsf.gov/biblio/10582382-constraint-conditioned-actor-critic-offline-safe-reinforcement-learning) [A]
- AnySearch: budget per question, sampled from an adaptive distribution, not relabeled — [arXiv 2609.00813](https://arxiv.org/html/2609.00813) [H]
- A 2025 trajectory-constrained diffusion paper adds cost-returns-to-go relabeling with data augmentation — [DOI](https://doi.org/10.65109/dhbe7400) [A]

### Inferences
- CCPO's data model is the main contrast for the claim: it still needs behavior policies for several thresholds, whereas relabeling needs none per limit. That contrast holds only against CCPO; SeqRoute and Brahmanage already do it offline for totals.
- Budgeted RL's next budget can exceed the current one on a single branch, since only E[β_a] ≤ β is required in exploration. A reader might mistake this for replenishment, but it is allocation by the policy, not a refill rate.

### Gaps
- Follow-ups citing these papers are under-covered; see the search log question below.
- Sauté's remark on "known safety state transitions" does not specify relabeling. Reading it that way is my inference.

## Remaining checklist: risk and CVaR conditioning, average-cost, instantaneous and cumulative, replenishing resources, networking

### Takeaway
None of these lines supplied a counterexample. Risk and CVaR conditioning use per-episode tail budgets or a risk level. Average-cost constraints bound the long-run rate with no burst control. Replenishing-budget work exists in bandits, control and character animation, but without a cost-limit input plus relabeling. In my search, networking RL treats the bucket as fixed infrastructure or as something to tune.

### Cited Findings
- **Risk-conditioned:** an agent can change its risk measure at run time "without expensive fine-tuning or retraining"; the title implies risk is an input, but I did not confirm it in the body — [Choi et al.](https://arxiv.org/abs/2104.03111) [A]
- **CVaR with budget in the state:** a residual tail budget augments the state, and exploration randomizes the initial risk budget — [Min, Predictive CVaR Q-learning](https://openreview.net/pdf/113d8715ed82ee45124b3c95bf34fbd0ad2475e6.pdf) [S]
- **CVaR distributional:** [Lim and Malik, NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/file/c88a2bd0e793550d0e885aa6e31ca277-Paper-Conference.pdf) [S]. Accumulated-cost augmentation with reward penalties: [arXiv 2301.11592](https://arxiv.org/html/2301.11592v2) [S]
- **Instantaneous and cumulative:** "An instantaneous constraint is a constraint that the chosen action needs to satisfy in each step"; cumulative constraints bound a sum or mean — [IPO](https://jhc.sjtu.edu.cn/~jiaxinding/media/ConstrainedRL_AAAI.pdf) [T]. No intermediate window type appeared in the passages read.
- **Average-cost:** deep RL with average-cost safety constraints is treated in [arXiv 2106.07329](https://arxiv.org/pdf/2106.07329) [T, relevant passages only]
- **Anytime constraints:** P[∀k, Σ_{t≤k} c_t ≤ B] = 1, optimal policies are "augmented with cumulative costs", and "Refueling stations are captured using negative costs" — [McMahan and Zhu](https://proceedings.mlr.press/v238/mcmahan24a/mcmahan24a.pdf) [T]
- **Reset-renewable budgets:** the budget "can be instantaneously reset to its maximum value by visiting a special part of the state space", with the state expanded by the budget. This is dynamic programming, not RL — [Takei et al.](https://ar5iv.labs.arxiv.org/html/1110.6221) [H]
- **Replenishing bandits:** resources can be replenished by a positive amount; this is a bandit model rather than an MDP — [Bernasconi et al., ICLR 2024](https://arxiv.org/pdf/2306.08470) [S]
- **Fatigue as a replenishing state in RL:** MF is in the observation space and randomized at each reset. R sets a sustained level ("20% MVC can be held indefinitely"). "We found that we can train on a single (F, R, r)-triplet but test on a variety of combinations". This is the nearest match on rate-style generalization, but it models fatigue limiting torque, with no cost limit input, window bound or relabeling — [Cheema et al.](https://arxiv.org/pdf/2310.08583) [T]
- **Burstable cloud instances:** unspent CPU credits accrue, and credits are spent to burst — [AWS docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-credits-baseline-concepts.html) [S]. I found no RL work controlling them.

### Inferences
- The Cheema paper is worth citing in a biped paper as the closest locomotion-adjacent relative. Its generalization runs through the observed fatigue state, not through a stated limit.
- Credits behave like a token bucket and LoRa duty cycles are window limits, so both domains could hold unpublished or hard-to-find matches.

### Gaps
- Choi, Min and Lim were not read in full. CVaR conditioning is only lightly checked.
- Fine-grained coverage of INFOCOM and SIGCOMM was not possible through open search; I saw only what the search tool and OpenAlex surfaced.

## Is anything from 2024 to 2026 close?

### Takeaway
Yes. 2026 is crowded with per-episode budget conditioning and relabeling, and with sliding-window constraints that lack conditioning. None of the 2024 to 2026 papers I read joins them. The risk of being scooped looks real: these lines are converging.

### Cited Findings
- March 2026: Brahmanage and Kumar, transition-level budget sampling, also at ICAPS 2026 — [arXiv](https://arxiv.org/abs/2603.22292) [T]; [ICAPS DOI](https://doi.org/10.1609/icaps.v36i1.42876) [A]
- May 2026: SDGD, sub-trajectory limits with a single conditioned model — [arXiv 2605.02777](https://arxiv.org/pdf/2605.02777) [T]
- May 2026: SeqRoute, Hindsight Budget Relabeling — [arXiv 2605.25424](https://arxiv.org/pdf/2605.25424) [T]
- June 2026: UCPO, stock as input, no retraining — [arXiv 2606.14029](https://arxiv.org/html/2606.14029) [H]
- July 2026 onward: sliding-window auto-bidding, version dated 3 Nov 2026 — [arXiv 2607.25233](https://arxiv.org/html/2607.25233) [H]
- September 2026: AnySearch, sampled budgets — [arXiv 2609.00813](https://arxiv.org/html/2609.00813) [H]
- October 2026: exact sliding-window bandits — [arXiv 2610.08745](https://arxiv.org/html/2610.08745) [H]
- February 2026: GAS, target relabeling — [arXiv 2602.05323](https://arxiv.org/html/2602.05323) [H]
- November 2025: virtual queue in a DQN state — [arXiv 2511.01406](https://arxiv.org/pdf/2511.01406) [T]

### Inferences
- A re-check just before submission is worthwhile, since three relevant preprints appeared in May and June 2026 alone.
- The name "Hindsight Budget Relabeling" is taken. Using it, or a near copy, requires citing SeqRoute.

### Gaps
- Preprints in the weeks before this search may not yet be indexed.

## What was searched, and what found nothing?

### Takeaway
Direct phrasing of the claim found nothing, in web search or OpenAlex. Those are null results from limited tools, not evidence that no such paper exists. The citation trail is thin because Semantic Scholar was rate-limited throughout.

### Cited Findings

Queries that found no paper with the rolling budget plus limit conditioning (web search tool):
- "token bucket constraint reinforcement learning policy budget state": LLM token-budget papers only
- "sliding window cost constraint safe reinforcement learning windowed CMDP": none
- "rate-limited reinforcement learning leaky bucket cost constraint replenishing budget": none
- "deep reinforcement learning token bucket traffic shaping state includes bucket occupancy": none directly; led to Chakrabarti
- "peak constraint moving average cost RL policy time window constraint CMDP robot thermal": none
- "replenishing safety budget ... refill recovers over time safe RL": none
- "Saute RL follow-up rolling OR windowed safety state budget cost constraint per window arXiv 2025": none
- "safe reinforcement learning sustained and burst cost limits rate budget conditioned policy rolling budget goal": none
- "constraint-conditioned OR threshold-conditioned ... cost over last T steps moving window limit policy input" (arXiv only): none
- "moving-average cost safe RL cost rate constraint exponential moving average...": none
- "rolling horizon constrained MDP ... rolling window budget regret online CMDP": none
- "relabel cost threshold off-policy replay safe RL threshold-conditioned critic hindsight constraint threshold": none combining these online
- "token bucket parameters input policy generalize multiple token bucket configurations...": patents and networking only
- "periodic budget reset every N steps cost constraint...": no safe RL paper; led to Takei
- "leaky integrator cost state augmentation ... exponential forgetting...": none
- "rate-constrained Markov decision process ... bucket capacity recursion policy conditioned on limit": none
- "duty cycle constraint sliding window reinforcement learning LoRaWAN...": RL papers exist; no rolling-budget state found
- "burstable cloud instances CPU credits reinforcement learning...": no RL work
- "adaptive token bucket parameters deep reinforcement learning rate limiter network INFOCOM...": no INFOCOM paper; led to arXiv 2511.03279
- "citing Constraint-Conditioned Policy Optimization ... single training run 2025": no direct citer surfaced

Queries that did produce relevant results:
- "hindsight relabeling cost limit budget constraint-conditioned policy safe RL": SeqRoute, SDGD, Brahmanage
- "constrained reinforcement learning with time window constraints ... every window": bandits and auto-bidding
- "token bucket reinforcement learning constraint offloading Lagrangian MDP policy rate burst": Chakrabarti
- "virtual queue as state input deep reinforcement learning...": Zakeri
- "energy harvesting reinforcement learning battery state...": Sharma
- "offline safe RL cost-to-go relabel data augmentation unseen cost limit...": CDT, GAS
- "fatigue recovery dynamics in state reinforcement learning...": Cheema

OpenAlex, title-and-abstract filter, with hit totals before my own keyword filtering. Totals are mostly noise; relevant survivors in brackets:
- "token bucket reinforcement learning" 13 (Cho 2020, the IJCNN 2024 paper, arXiv 2511.03279, older traffic shapers)
- "token bucket deep Q network" 4 (Qiu 2022)
- "leaky bucket reinforcement learning" 3 (the usage parameter control chapter)
- "sliding window constrained reinforcement learning" 18 (the UAV paper only)
- "sliding window constrained Markov decision process" 6 (none new)
- "windowed constrained Markov decision process" 58 (none relevant)
- "constraint relabeling reinforcement learning" 10 (SDGD, GAS, the 2025 cost-return relabeling paper)
- "replenish safety budget reinforcement learning" 1 (none)
- "burst budget safe reinforcement learning" 2 (none)
- "moving average cost constraint reinforcement learning" 14 (none)
- "cost per window constraint reinforcement learning" 16 (none; vehicle routing time windows)
- "arrival curve reinforcement learning" 27 (none combining arrival curves with a limit-conditioned policy)
- "rate limit offloading deep reinforcement learning token" 0

OpenAlex full-text style phrases, with hits checked only against title and abstract:
- "hindsight budget" 3 and "budget relabeling" 3: SeqRoute was the only relevant hit
- "rolling budget" 11, "rolling window" with constrained RL 25, "moving window" with constrained RL 14: none matched in title or abstract

Citation trail through OpenAlex: CCPO 7 citers, Sauté 22, Simmer 8 and 1 on two records, Budgeted RL 12. That is 44 unique citers scanned by keyword in title and abstract for window, rolling, token bucket, relabel, replenish and similar terms. The only keyword hit was an irrelevant neuroscience-style paper on "peaks" — [OpenAlex](https://api.openalex.org/works/doi:10.48550/arXiv.2310.03718)

Tool failures:
- Semantic Scholar citations API returned HTTP 429 on every attempt, through both shell and fetch, for all four named papers.
- The arXiv export API returned an empty response.
- The ETRI copy of RACER refused the connection.
- The NSF PDF of CCAC would not convert to text, and OpenReview returned a verification page.

### Inferences
- OpenAlex citer counts look far too low for papers of this age and profile, so the citation trail should be treated as a small sample, not as coverage.
- Keyword filtering on abstracts cannot catch a paper that uses a rolling budget without saying so in its abstract.

### Gaps
- Before submission, someone should run Google Scholar "cited by" on CCPO, Sauté and Qiu 2022 by hand, and check citers of SeqRoute and Brahmanage and Kumar.
- Unchecked leads:
  - Liakopoulos et al. 2019, "Cautious regret minimization" (windowed budgets, cited by [arXiv 2610.08745](https://arxiv.org/html/2610.08745))
  - Abdullah et al. 2026, constrained online optimization with memory (same source)
  - Murphy, "Bypassing Duty-Cycle Limitations for RL-Enhanced LoRaWAN Communications" (seen in a search summary only)
  - [State-augmented wireless coexistence, arXiv 2506.22652](https://arxiv.org/pdf/2506.22652) [S]
  - [Multiple mean-payoff under local stability, arXiv 2412.13369](https://arxiv.org/html/2412.13369v1) [S]
