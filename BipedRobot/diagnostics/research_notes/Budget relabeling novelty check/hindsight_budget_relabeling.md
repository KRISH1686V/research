# Hindsight relabeling of cost budgets, cost limits and constraint thresholds in RL: novelty check

Claim under test: rolling (token-bucket, sliding-window) gait budgets on flight, stance slip and torque saturation are treated as hindsight goals. They are relabeled together with the velocity command during online off-policy training (SAC + HER), and bucket levels are recomputed exactly under the relabeled budget, so one run covers many gait limits on a low-cost 12-DOF servo biped.

Method note. Unless a line says otherwise, "verified" means I read the passage in the paper's own text. I downloaded each arXiv PDF, extracted it with pdftotext and read or grepped the relevant section, or I read the arXiv HTML. Quotes are copied from that text; pdftotext sometimes drops symbols in formulas. Lines marked "(full-text fetch)" were read through a fetch tool that returned quotes from the publisher's full-text page (PMC), not from my own extraction. Items under "Inferences" are my reasoning, not statements in the sources.

## Q1. Which papers relabel a budget, cost limit or constraint threshold in hindsight, and how exactly?

### Takeaway
Hindsight-style relabeling of a budget or cost limit is **not new**. At least six earlier papers relabel a cost limit, budget or constraint target after the fact:
- CHER (2019)
- SaFormer (2023)
- CDT (2023)
- HALO (2025)
- BCRL (2026)
- SeqRoute (2026)

GAS (2026) also relabels cost return-to-go targets. SaFormer is the most dangerous for the claim. It relabels the cost-limit token with the realized cost return during online fine-tuning, recomputes cost-to-go along the trajectory, and its cost is cumulative joint torque on MuJoCo locomotion.

None of the papers found relabels a **rolling, refilling or windowed** budget. None relabels a budget **jointly with a goal or command**. None does it in **online off-policy actor-critic (SAC/HER-style) training**, and none does it on **legged hardware**.

### Cited Findings

**Comparison table (verified from each paper's text):**

| Paper | What is relabeled | Budget type | Online / offline | Balance recomputed along trajectory? | Joint with goal/command? | Domain / hardware | Learner |
|---|---|---|---|---|---|---|---|
| CHER, Wang et al., CIKM 2019 | State-level constraint target c_s, swapped for every c_s' in a finite set | Per-request (per-state) target, not a budget | Trained on replayed logs plus a production A/B test | No; there is no balance | No | Taobao ad exposure; no robot | DDPG + constraint loss |
| SaFormer, Zhang et al., 2023 | Cost-limit token D, plus cost-to-go and return-to-go | Per-episode total (cumulative torque) | Offline, plus an online fine-tuning loop | Yes: Ĉ_t = Σ_{t'≥t} c_t', and D = Ĉ_0 | No | D4RL Hopper, Walker2d, HalfCheetah; sim only | Transformer + critic |
| CDT, Liu et al., ICML 2023 | Cost and reward return-to-go sequences of a Pareto trajectory | Per-episode total | Offline | Shifted element-wise by κ − C(τ*) | No | Safety-Gym-style sims | Decision transformer |
| HALO, Dong et al., 2025 | Budget B replaced by the realized cost | Per-day budget (+ ROI target) | Offline dataset (AuctionNet) | Per-suffix realized cost Ĉ_τT for every τ | No | Ad auto-bidding; no robot | Supervised B-spline meta-model |
| BCRL, Brahmanage & Kumar, ICAPS 2026 | Remaining budget δ resampled for each transition | Per-episode discounted budget carried as a state | Offline | One-step transition δ' = g(s,a,s',δ) only | No | DSRL, grid worlds, maritime; no hardware | IQL |
| GAS, Liu et al., 2026 | Segment return-to-go and cost-to-go | Segment-level | Offline | Sampled: Ĉ ~ U(C_t:Γ, C_max) | No (its "goals" are reward-cost targets) | DSRL | Goal functions + generative policy |
| SeqRoute, Xu et al., 2026 | Initial budget B0 (five virtual values) | Per-session total | Offline | Yes, exact replay: b_{t+1} = b_t − C/B_total, with a bankruptcy terminal | No | LLM routing; vLLM servers only | Discrete CQL |
| QCP, Kang et al., Oct 2026 | Trajectory quality label from the realized score | Per-trajectory score (speed, jerk, path length, tilt); not a budget | Iterative online rollouts | Quantile bins recomputed over the whole dataset | No | Robomimic, RoboEval; real Franka FR3 | Flow-matching behavior cloning, retrained from scratch |

**Per-paper detail:**

- **CHER, "Constrained Hindsight Experience Replay" (Wang et al., CIKM 2019).**
  - Earliest paper found that names a "hindsight" mechanism for constraints. Its Algorithm 2 stores a transition (s_t, a_t, r_t, s_{t+1}, c_s), and then "for c_s′ ∈ C_S use transition (s_t, a_t, r_t, s_{t+1}, c_s′) to train π_b(s_t; c_s′) by L′ = L_RL + L_cs′". — [arXiv 1809.03149](https://arxiv.org/abs/1809.03149)
  - "Different from HER, CHER does not directly revise the reward. Instead, it uses different constraints to define the extra loss L_CS during training." The constraint enters as a per-state penalty, for example "δ(s_t, a_t, s_{t+1}|C_s) = −(pvr_t − 0.4)²". — [arXiv 1809.03149](https://arxiv.org/abs/1809.03149)
  - Constraints are a request-level ad-exposure ratio and a platform-level daily ratio. Training replays "the users' access logs in chronological order to simulate the users' requests", and the paper also reports "production A/B test experiments". — [arXiv 1809.03149](https://arxiv.org/abs/1809.03149)
  - There is no accumulated balance, no goal and no robot.
- **SaFormer (Zhang et al., arXiv Jan 2023).**
  - Abstract: "a two-stage auto-regression conditioned by the maximum remaining cost". It treats "the constraint as contextual tokens". — [arXiv 2301.12203](https://arxiv.org/abs/2301.12203)
  - On the data: "We customize safety constraints hind-sightly and relabel the dataset with specific cost criteria", and "the existing offline data is relabeled hind-sightly according to certain cost criteria". — [arXiv 2301.12203](https://arxiv.org/abs/2301.12203)
  - Algorithm 3, "SaFormer Online Fine-tuning":
    1. "Set d = αd as the target cost limit."
    2. "Rollout trajectory τ".
    3. "Relabel τ as R̂_t = Σ_{t'=t}^T r_t', Ĉ_t = Σ_{t'=t}^T c_t', D = Ĉ_0".
    4. "Aggregate the offline data".
    5. Fine-tune.

    The text adds: "we use the actual cost return as the value of token D when we relabel the newly collected trajectory." — [arXiv 2301.12203](https://arxiv.org/abs/2301.12203)
  - Cost definition: "Our safety consideration is to prolong the lifespan of motors. Therefore, we specify the cost as c_t = Σ_{i=1}^M |a_t^i| standing for the total torque applied to the M joints", used on D4RL Hopper, Walker2d and HalfCheetah. Results are simulation only. — [arXiv 2301.12203](https://arxiv.org/abs/2301.12203)
- **CDT, "Data Augmentation via Relabeling" (Liu et al., ICML 2023).**
  - Algorithm 1:
    1. "κ_i ∼ Uniform(c_min, c_max)", plus a reward target "ρ_i ∼ Uniform(RF(κ_i, T), r_max)".
    2. "τ*_i ← argmax R(τ), s.t. C(τ) ≤ κ_i".
    3. "τ̂_i ← {R*_i + ρ_i − R(τ*_i), C*_i + κ_i − C(τ*_i), s*_i, a*_i}".
  - "The intuition is to relabel the associated Pareto trajectory's reward and cost returns, such that the agent can learn to imitate the behavior of the most rewarding and safe trajectory τ* when the desired return (ρ, κ) is infeasible". Offline, per-episode totals. — [arXiv 2302.07351](https://arxiv.org/abs/2302.07351)
  - The same CDT algorithm is reproduced as "Algorithm 2 Data Augmentation via Relabeling" in the Q-learning Penalized Transformer appendix. — [arXiv 2609.34426](https://arxiv.org/abs/2609.34426)
- **HALO (Shopee, arXiv Aug 2025).**
  - "replacing B with the realized cost C_realized(β) transforms the trajectory into an optimal exploration for the posterior constraint configuration B′ = C_realized(β). Through this mechanism, every exploratory trial becomes an effective sample for some budget-constrained scenario." It calls this "the Hindsight Mechanism, which establishes a bijection between historical explorations and valid constraint configurations". — [arXiv 2508.03267](https://arxiv.org/abs/2508.03267)
  - Algorithm 1 records, for every start step τ, the realized suffix cost and value H_τ = (s_τ, β̂_τ, Ĉ_τT, V̂_τT). That amounts to recomputing the remaining budget for every suffix. — [arXiv 2508.03267](https://arxiv.org/abs/2508.03267)
  - Learning is supervised: a "Dual B-Spline Policy" whose MetaModel is trained against "ground-truth optimal labels (obtained by Algo. 1)". Evaluation uses AuctionNet, with "a 14-day training set (678 bidding trajectories) and a 7-day test set". — [arXiv 2508.03267](https://arxiv.org/abs/2508.03267)
- **BCRL, Budget-Conditioned Reachability (Brahmanage & Kumar, ICAPS 2026).**
  - Algorithm steps: "δ ∼ U[Q_C(s,a), δ_max] ▷ sample budget from uniform dist.", then "δ′ = g(s,a,s′,δ) ▷ compute next budget". — [arXiv 2603.22292](https://arxiv.org/abs/2603.22292); [ICAPS listing](https://ojs.aaai.org/index.php/ICAPS/article/view/42876)
  - "Unlike a fixed dataset, this augmented dataset is not pregenerated; instead, we sample mini-batches from it dynamically during training … Crucially, this process acts as data augmentation rather than filtering … The rewards r(s,a) and c(s,a) remain unchanged". The learner is IQL and the setting is offline. The paper says Saute RL "requires online rollouts to track budgets". — [arXiv 2603.22292](https://arxiv.org/html/2603.22292v2)
- **GAS (Liu et al., arXiv Feb 2026).**
  - "Transition-level Return Relabeling": "R̂_t:Γ = U((1 − δ)R_t:Γ, (1 + δ)R_t:Γ), Ĉ_t:Γ = U(C_t:Γ, C_max)".
  - "GAS does not directly update the policy guided by the relabeled values. Instead, we utilize them to assist training through intermediate optimal goals". Offline (DSRL). — [arXiv 2602.05323](https://arxiv.org/abs/2602.05323)
- **SeqRoute (Xu, Zheng, Wang, arXiv May 2026).** See Q2 for the full mechanism. — [arXiv 2605.25424](https://arxiv.org/abs/2605.25424)
- **QCP (Kang et al., arXiv 27 Oct 2026).** See Q2. QCP relabels a trajectory-quality attribute, not a budget. — [arXiv 2610.37393](https://arxiv.org/abs/2610.37393)

**Budget-conditioned methods that sample budgets but do not relabel them in hindsight (verified):**

- **Sauté RL.** "Since the remaining safety budget is the initial state, we can randomly sample different safety budgets at the beginning of the episode." In experiments the initial safety state is "uniformly sampled from the interval [5, 100]". The budget is chosen at reset; nothing is relabeled afterwards. — [arXiv 2202.06558](https://arxiv.org/abs/2202.06558)
- **Simmer.** Schedules the initial safety budget during training, with no relabeling. — [arXiv 2206.02675](https://arxiv.org/abs/2206.02675)
- **Budgeted RL / BFTQ (Carrara et al., NeurIPS 2019).**
  - The state and action are augmented with budgets ((s, β), (a, β_a)).
  - For exploration, "we also need to sample the initial budget β0. Importantly, we pick a uniform distribution P(β0) = U(B)". — [arXiv 1903.01004](https://arxiv.org/abs/1903.01004)
- **CCPO (Yao et al., NeurIPS 2023).**
  - Trains "behavior agents, each under their respective target constraint" and uses Versatile Value Estimation to approximate Q for unseen thresholds.
  - A grep of its full text for relabel, hindsight and replay found only reference-list hits. — [arXiv 2310.03718](https://arxiv.org/abs/2310.03718)
- **TREBI.** Real-time budget constraints in offline safe RL, solved by diffusion planning. Its text has no relabel or hindsight passage, only a cited title. — [arXiv 2306.00603](https://arxiv.org/abs/2306.00603)
- **SDGD (arXiv May 2026).** Feasible Trajectory Relabeling relabels the **reward** target, not the budget: "R̂(τ) = R(τ) + r_us h_f(τ)", where h_f flags cost in the executable prefix. "FTR modifies the target used to train the reward predictor, while leaving the diffusion model and the cost limit conditioning unchanged." Offline; cost limits may change within an episode at deployment. — [arXiv 2605.02777](https://arxiv.org/abs/2605.02777)
- **DSRL benchmark.** Its noise filter "select[s] a subset of high-cost trajectories and relabel[s] their cost returns as low" to inject outliers. This is corruption, not hindsight. — [arXiv 2306.09303](https://arxiv.org/abs/2306.09303)
- **Temporal-logic-conditioned DT (SDT).** It relabels per-step cost into a **windowed** STL-violation cost, "c′_t = 1 if c_pt = 1 or [c_v(t−5), …, c_v(t−1)] = 1". This is a windowed cost definition for one fixed specification, offline, with no budget relabeling. — [arXiv 2402.17217](https://arxiv.org/abs/2402.17217)

### Inferences
- The relabeling styles in prior work fall into three groups:
  - Virtual-budget sweeps: SeqRoute's five fixed B0 values, BCRL's uniform δ, CDT's uniform κ, CHER's finite constraint set.
  - Achieved-cost substitution, the closest HER analogue: SaFormer's D = Ĉ_0, HALO's B′ = realized cost, QCP's relabel by realized score.
  - Target perturbation: GAS.

  The claim should state which style it uses. "Budget as a hindsight goal" reads like the achieved-cost style, and that is exactly SaFormer and HALO.
- "Exact recomputation of the balance under the relabeled budget" has a precedent for **total** budgets. SeqRoute replays the same actions through a deterministic budget transition and re-derives the bankruptcy termination, and SaFormer and HALO recompute cost-to-go from realized costs.
  - The part that still looks new is path-dependent recomputation for a **refilling or windowed** budget. With a capped token bucket, the level at step t depends on the whole history and on the bucket parameters, so changing the budget changes every later level and every violation flag. Nothing found does that.
- CHER is the only earlier work found that uses the HER name for constraints in an online-style off-policy loop. Its "constraint" is a per-state target with no accumulation, so it does not cover budgets.

### Gaps
- Budgeted RL (Carrara et al.): I did not confirm whether BFTQ also resamples β for transitions already in the batch, as opposed to only sampling β0 at exploration time. Only the exploration passage was read.
- CCAC (Guo et al., ICLR 2025, "Constraint-Conditioned Actor-Critic for Offline Safe RL") could not be read: NSF PAR returned 403 and the OpenReview PDF did not download. Its threshold-sampling and augmentation mechanism is unverified. It needs checking before submission. — [ICLR listing](https://iclr.cc/virtual/2025/poster/28392)
- SaFormer's fine-tuning objective is MaxEnt-flavoured ("we reformulate the naive NLL objective … under the MaxEnt RL framework (Haarnoja et al., 2018)"). I did not check how much of an off-policy critic is involved beyond its "critic ζ". — [arXiv 2301.12203](https://arxiv.org/abs/2301.12203)

## Q2. The named papers, checked against their own text

### Takeaway
All seven named papers exist and say what was expected, with corrections for QCP and BCRL:
- SeqRoute is the cleanest budget relabeling, but it is offline, uses a total budget and has no goal.
- HPR and the coverage-collapse paper relabel preferences, not budgets.
- Generalized Hindsight relabels reward-task parameters. These include goal position plus obstacle and action-penalty weights, but no budgets.
- RbSL relabels goals only.
- "BCRL" is Budget-Conditioned Reachability (ICAPS 2026).
- "QCP" is the Quality-Controllable Policy (arXiv 2610.37393, Oct 2026). It is not a cost-constrained policy method.

### Cited Findings
- **SeqRoute (arXiv 2605.25424, 25 May 2026; Xu, Zheng, Wang).**
  - "Inspired by Hindsight Experience Replay [1], HBR retrospectively simulates recorded trajectories under a spectrum of hypothetical initial budgets."
  - "For each trajectory τ = {(h_t, a_t, y_t, q_t+1)} in D_raw and each hypothetical budget B0, HBR constructs a new trajectory by replaying the same (h_t, a_t, y_t) sequence through the deterministic budget transition (Eq. 3). At each turn, if b_t+1 ≤ 0, the transition is assigned reward r_t = −η and the episode terminates."
  - "By sweeping five virtual budgets {500, 1500, 3000, 5000, 8000}, each raw session is projected into five budget-conditioned trajectories, expanding the dataset to over 2.38 million transitions."
  - Proposition 1 justifies this because the environment is "conditionally independent of the artificial budget state"; "the budget transition is a bookkeeping operation".
  - The budget is a per-session total (b_t ∈ [0,1], b_1 = 1). The setting is offline ("from historical logs via offline RL"), the learner is discrete CQL, and there is no goal or command conditioning. The domain is LLM routing between an 8B and a 70B model, with no physical hardware.
  - Minor internal inconsistency: Eq. 3 normalises by B_total, while Appendix B says "decrementing the budget by the actual token cost".
  - — [arXiv 2605.25424](https://arxiv.org/abs/2605.25424); [HTML](https://arxiv.org/html/2605.25424)
- **Hindsight Preference Replay (arXiv 2601.11604, Jan 2026; Shianifar, Schukat, Mason).**
  - It "retroactively relabels stored transitions with alternative preferences", sampled from "a Dirichlet distribution centered around the original"; "The reward vector r is left unchanged".
  - Built on CAPQL, which "extends the Soft Actor-Critic (SAC) framework". Online off-policy, on MO-Gymnasium MuJoCo tasks.
  - "Constraint" does not appear in the text (0 matches). The "budget" matches refer to training-step budgets.
  - — [arXiv 2601.11604](https://arxiv.org/abs/2601.11604)
- **On Preference Coverage Collapse from Hindsight Relabeling (arXiv 2609.26918).**
  - It relabels "with the preference direction the agent achieved rather than the one asked for", via "w′ = normalize(clip(ô, 0, ∞))".
  - It flags that "not every objective contains a natural notion of achievement in the way a goal is achieved (a cost objective, for instance, is never reached so much as minimized)", and that the clip is "silently ill-posed for objectives that are negative by construction".
  - Its fix is her_mix with λ = 0.25. Simulation only (MO-Gymnasium); no budgets or thresholds relabeled.
  - — [arXiv 2609.26918](https://arxiv.org/abs/2609.26918)
- **Generalized Hindsight (arXiv 2002.11708; Li, Pinto, Abbeel, NeurIPS 2020).**
  - Relabels the task variable z with "Approximate IRL Relabeling (AIR)" or an advantage-based strategy, using SAC.
  - In PointReacher, "The 6-dimensional task vector is z = (x_g, y_g, x_obst, y_obst, u, v), where u and v control the weighting between the goal rewards, obstacle rewards, and action magnitude penalty". The environments add "dense reward parameterization for practical aspects of manipulation like energy consumption … and safety".
  - The words "constraint", "budget", "cost limit" and "threshold" each appear 0 times in the extracted text.
  - — [arXiv 2002.11708](https://arxiv.org/abs/2002.11708)
- **RbSL (arXiv 2403.01734, ICRA 2024; Cao et al.).**
  - "we relabel the goal with a future state g′ = ϕ(s_t′) … In an environment with constraints, we add the information of constraints as a part of the state", and the relabel applies only when "s_T ∉ o", where o is the unsafe area.
  - A separate cost-shaping step (Eq. 11) is used for the recovery policy. No cost limit or budget is relabeled.
  - Offline, with a real Franka Emika Panda deployment.
  - — [arXiv 2403.01734](https://arxiv.org/abs/2403.01734)
- **BCRL = "Beyond Hard Constraints: Budget-Conditioned Reachability for Safe Offline RL" (Brahmanage & Kumar, ICAPS 2026; code github.com/janakact/bcrl).**
  - Per-transition budget resampling δ ∼ U[Q*_C(s,a), δ_max] with a one-step budget transition, offline IQL. See Q1 for the quotes.
  - It does not use the word hindsight. It calls the mechanism "data augmentation".
  - — [arXiv 2603.22292](https://arxiv.org/abs/2603.22292)
- **QCP = "Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling" (Kang et al., KAIST/LGE/Microsoft Research, arXiv 27 Oct 2026).**
  - A rule-based score m(τ) is quantized "into B equal-frequency bins" with labels q(τ) from the empirical CDF. Scores include 1/T for speed, "action jerk", "total joint path length", and "end-effector tilt from vertical".
  - Algorithm 1:
    1. "Collect rollouts with π_θk(· | h_t, q⋆)".
    2. "Append successful rollouts".
    3. "Recompute q(τ) for all τ ∈ D_k+1".
    4. "Train π_θk+1 from scratch".
  - "We therefore relabel each successful rollout in hindsight by its realized score m(τ) rather than the requested label."
  - The learner is flow-matching behavior cloning (also a π0.5 VLA). There is real-robot validation on a Franka Research 3 for conveyor speed and wine-glass transport stability. There is no constraint, budget or cost limit, and no RL value learning; RL methods appear only as baselines.
  - — [arXiv 2610.37393](https://arxiv.org/abs/2610.37393)

### Inferences
- QCP matters to the claim's framing, not its mechanism. It shows a hindsight-relabeling loop on a trajectory-level "how well did it execute" attribute, with online collection and a real robot, published only a week or two before this check. Reviewers may cite it next to SaFormer.
- Generalized Hindsight already relabels a goal together with penalty weights for obstacle and action magnitude. A reviewer could call that "goal + soft-constraint relabeling", so the paper should explain that weights are not limits or budgets.

### Gaps
- Where I quote formulas from pdftotext output, some symbols may be lost. The SeqRoute HTML and PDF wording matched each other.

## Q3. Offline safe RL cost-return relabeling, multi-objective preference relabeling, and joint goal + constraint relabeling

### Takeaway
- In offline safe RL, relabeling or augmenting cost returns is an established technique: CDT 2023, SaFormer 2023, GAS 2026, SDT 2024, plus QPT reusing CDT's algorithm. All of it is offline sequence or generative modelling with per-episode or per-segment totals.
- In multi-objective RL, HER-style preference relabeling is old: Envelope MOQ 2019, LP3 2021, HPR 2026. LP3 is the closest. It relabels Lagrange-like preference weights on replay samples in constrained humanoid and walker tasks, but the thresholds stay fixed.
- Joint relabeling of a goal together with a constraint-like quantity exists in three forms, but never with a cost budget:
  - Goal plus avoid region: RADT.
  - Goal plus horizon: TDM.
  - Command (return plus horizon) in hindsight: UDRL.

### Cited Findings
- **Envelope MOQ (Yang, Sun, Narasimhan, NeurIPS 2019).** "for each sampled transition record … from replay buffer D_τ, we associate it with N_ω preferences {ω_1, …, ω_Nω} sampled from D_ω … In this way, the trajectories can be replayed with arbitrary preferences with 'hindsight'." — [arXiv 1908.08342](https://arxiv.org/abs/1908.08342)
- **LP3 (Huang et al., DeepMind, CoRL 2021).**
  - For off-policy evaluation: "For each learning step we sample L transitions from the replay buffer … Motivated by prior work on learning from off-policy data [33 = HER], we use relabeling to augment the states with preferences sampled from the current preference distribution ε_i ∼ p_old(ε)". Lagrangian relaxation corresponds to "the preference ε = [1, λ_1, …, λ_K]".
  - Constraints are fixed thresholds β_k. For walker run and humanoid walk, "we set inequality constraints on moving and energy usage". It "has not been validated yet on a real robot platform".
  - — [PMLR v164](https://proceedings.mlr.press/v164/huang22a.html)
- **UDRL (Srivastava et al., 2019).** Training pairs are built by sampling a segment and computing "the values of d_r and d_h for it in hindsight", where d_r is the desired return and d_h the desired horizon. — [arXiv 1912.02877](https://arxiv.org/abs/1912.02877)
- **TDM (Pong et al., ICLR 2018).**
  - "TDMs can benefit from arbitrary relabeling of the goal states g and the horizon τ … This relabeling enables simultaneous, data-efficient learning of short-horizon and long-horizon behaviors". The algorithm line reads "Relabel time horizons and goal states τ_m, s_g,m", with targets bootstrapped at τ_m − 1.
  - Online off-policy, with a real 7-DoF Sawyer reaching test.
  - — [arXiv 1802.09081](https://arxiv.org/abs/1802.09081)
- **RADT (Li & Zitnik, arXiv 2505.19337, 2025).**
  - "we relabel the last state s_T as the goal state g in hindsight. Additionally, we can also relabel avoid regions in hindsight, a novel strategy that gets rid of the need to use a cost function".
  - Avoid-box centroids and widths are sampled randomly, and a binary avoid-success token is flipped with a second pass (full-text fetch).
  - Offline; FetchReachObstacle, MazeObstacle and a simulated cell-reprogramming task; no hardware.
  - — [arXiv 2505.19337](https://arxiv.org/abs/2505.19337); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12148088/)
- **Reward machines (CRM; Toro Icarte et al., JAIR 2022).** "CRM relabels experiences using the RM and HER uses goal states". CRM generates counterfactual experiences for every machine state; "it is unclear how to … learn a policy that generalizes to unseen RMs". — [arXiv 2010.03950](https://arxiv.org/abs/2010.03950)
- **LTL counterfactual experience replay (LCER; Voloshin et al., ICML 2023).** Generates "multiple counterfactual trajectories from a single on-policy trajectory" by recomputing automaton states. — [arXiv 2303.02135](https://arxiv.org/abs/2303.02135)
- **Goal-conditioned safe RL (Feng, Parimi, Williams, 2025).**
  - "the cost function and cost limit is task-dependent and is not applicable to the goal-conditioned setting where each goal is akin to a distinct task", so they "redefine the cost limit as a hyper-parameter, a soft constraint".
  - Lagrangian actor-critic. No budget relabeling found by grep (relabel, hindsight, HER).
  - — [arXiv 2502.17813](https://arxiv.org/abs/2502.17813)
- **Risk-bounded multi-agent visual navigation (Parimi & Williams, ICAPS 2026).** Allocates per-agent risk budgets δ_i during conflict-based search over a learned graph. That is planning-level budget allocation; the grep found no relabeling. — [arXiv 2509.08157](https://arxiv.org/abs/2509.08157)
- **SG-Safe (2304.09119).** Combines hindsight goal generation with a multi-objective sparse reward for collision avoidance in manipulation. Its relabeling is over goals only; no budget relabeling found. — [arXiv 2304.09119](https://arxiv.org/abs/2304.09119)
- **CHRL (Li et al., Scientific Reports 2024; arXiv 2310.15583)**, the closest **locomotion** HER precedent (full-text fetch).
  - Relabels the **velocity command**: "we sample a new goal for each transition", within ±0.3 linear velocity and ±0.5 yaw of the original, limited to the current curriculum range.
  - Rewards are recomputed from stored robot quantities (power, torques, foot contact, and others). The learner is REDQ, online, in PyBullet.
  - Deployed zero-shot on a ~15 kg, 12-actuated-joint quadruped.
  - No constraints or budgets are relabeled.
  - — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564515/); [arXiv 2310.15583](https://arxiv.org/abs/2310.15583)
- **LEMUR (arXiv 2607.29559).** Matched the full-text phrase "budget relabeling", but what it relabels is *vector rewards* after reward-model updates ("all transitions stored in the buffer are relabeled"). It does not relabel budgets. — [arXiv 2607.29559](https://arxiv.org/abs/2607.29559)

### Inferences
- The closest analogue for "relabel a goal or command **and** a countdown quantity jointly in online off-policy learning, with the target recomputed" is TDM, where the countdown is the horizon. The second closest is CHRL, which relabels the command in legged locomotion with an off-policy learner. Combining CHRL-style command relabeling with SeqRoute-style exact budget recomputation would be close to the claim, but no paper found does both.
- Reward-machine counterfactuals (CRM, LCER) are the best conceptual precedent for "recompute an auxiliary finite-state tracker along the stored trajectory". A token bucket is a counter-valued tracker. Reviewers from formal-methods RL may draw that link, so it is worth citing.
- LP3 is the closest **constrained-locomotion off-policy relabeling** work, with an energy-usage constraint on humanoid walk. What it relabels is preference or multiplier weights, not budget values, and its thresholds are fixed per run.

### Gaps
- PCN (Pareto Conditioned Networks) conditions on a desired return vector and horizon, but the grep found no "hindsight" or "relabel" wording. I did not trace how its training targets are formed. — [arXiv 2204.05036](https://arxiv.org/abs/2204.05036)
- SCARED (Safe In-Context RL, 2509.25582) cites "Guan et al. (2024) … cost-aware context encoder that uses supervised cost relabeling". I did not identify or read Guan et al. 2024. — [arXiv 2509.25582](https://arxiv.org/abs/2509.25582)
- I did not obtain the full text of two OpenAlex hits, both MDPI, both 2026: "Enhancing Safe Exploration Through Subgoal Guidance" and "Imagine to Ensure Safety in Hierarchical RL". The PDFs did not download and the HTML was over the fetch size limit. Each matched "hindsight experience replay" together with "cost limit" or "safety budget" in OpenAlex full-text search. — [DOI 10.3390/technologies14030146](https://doi.org/10.3390/technologies14030146); [DOI 10.3390/make8080243](https://doi.org/10.3390/make8080243)
- "Safe RL for Arm Manipulation with CMDP" (Robotics 2024), which matched "hindsight experience replay" together with "cost limit" in OpenAlex full text, was not read. — [DOI 10.3390/robotics13040063](https://doi.org/10.3390/robotics13040063)

## Q4. Is anything from 2024 to 2026 close to the full claim, and what did the search cover?

### Takeaway
**No paper found matches the full claim.** No paper found relabels a rolling (token-bucket or sliding-window) budget, recomputes bucket levels under the relabeled budget, relabels it jointly with a velocity command or goal, or does this in online off-policy training on legged hardware.

The 2024 to 2026 papers closest on single components are:
- SeqRoute 2026: exact balance recomputation, but offline and with a total budget.
- HALO 2025: realized-cost budget substitution.
- BCRL 2026: per-transition budget resampling.
- GAS 2026 and SDGD 2026: offline cost-return relabeling.
- QCP Oct 2026: online iterative hindsight relabeling of trajectory-quality attributes on a real robot.
- RADT 2025: joint goal + avoid-region relabeling, offline.
- CHRL 2024: velocity-command HER in legged locomotion.

Together with the older SaFormer 2023 and CHER 2019, these rule out any claim to be the first to relabel a budget or cost limit in hindsight.

### Cited Findings
- No paper turned up for rolling, windowed or refilling budgets with relabeling. Related items:
  - SDT uses a 5-step window inside a relabeled per-step cost, with no budget. — [arXiv 2402.17217](https://arxiv.org/abs/2402.17217)
  - A token-bucket edge-offloading paper keeps the bucket level as decision state, with no relabeling. — [arXiv 2010.13737](https://arxiv.org/abs/2010.13737)
  - A UAV safe-RL paper uses a "sliding-window baseline" for the advantage, with an "episodic cost budget d". — [arXiv 2508.09971](https://arxiv.org/abs/2508.09971)
- Low-cost legged hardware papers found have no hindsight or budgets. For example, a "$300 quadruped" (Mini Pupper 2) is trained with PPO in Isaac Lab, sim-to-real, with no constraint formulation. — [arXiv 2607.26434](https://arxiv.org/abs/2607.26434)
- Energy-constrained humanoid walking treats energy as a constraint with a fixed per-episode threshold; its text has no relabel or hindsight terms. — [arXiv 2602.06445](https://arxiv.org/abs/2602.06445)

### Search coverage (queries run; "nothing" means no relevant hit and is not proof of absence)

The arXiv API searches abstracts with exact phrases. OpenAlex full-text search covers only works with indexed full text, so it has partial coverage. I listed and screened results by title, and read the full text of candidates as noted above.

**arXiv API, exact phrase in abstract:**

| Query | Result |
|---|---|
| "hindsight" AND "budget" (cs.LG/AI/RO/stat.ML) | 40 results; relevant: SeqRoute, HALO, HPR |
| "relabeling", "relabelling" or "relabel" AND "budget" | SeqRoute, SDGD, HPR; the rest irrelevant |
| "relabeling" AND "cost limit" | SDGD only |
| "hindsight experience replay" AND "constraint" | CHER, and 2510.10332 (SAC + HER maneuvering; HER on goals only) |
| "relabeling" AND "safe reinforcement learning" | SDGD, GAS |
| "relabeling" AND "constrained" AND "reinforcement learning" | SeqRoute, GAS, SDGD, 2610.05446 (time-aware subgoal bootstrapping; not budgets) |
| "budget relabeling"; "hindsight budget" | SeqRoute only |
| "cost return" AND "relabel" | GAS only |
| "threshold-conditioned" AND RL | CCPO |
| "budget-conditioned" AND RL | 6 results; four of them downloaded and grepped, no relabel/hindsight |
| "constraint-conditioned" AND RL | 9 results, none about relabeling, by title |
| "constrained hindsight" | CHER |
| "avoid" AND "hindsight relabeling" | RADT, PIPER, an underwater navigation paper |
| "hindsight" AND "quadruped" | CHRL, and an obstacle-avoidance paper |
| "hindsight" AND "humanoid" AND RL | HPR, SceneBot (not read) |
| "token bucket" AND RL | 2 results (microservice rate limiting, video ordering); not read beyond titles |
| "sliding window" AND "constrained" AND RL AND "cost" | 2508.09971, and an irrelevant LLM sampling paper |
| "counterfactual budget"; "budget augmentation" AND RL; "QCP" AND RL; "BCRL" | Irrelevant only |

These arXiv phrase queries returned **nothing**:
- "relabeling" AND "cost threshold"
- "relabeling" AND "constraint threshold"
- "hindsight" AND "cost limit"
- "hindsight" AND "cost threshold"
- "hindsight" AND "constrained reinforcement learning"
- "hindsight" AND "safe reinforcement learning"
- "hindsight experience replay" AND "safety"
- "constraint relabeling"
- "cost relabeling"
- "hindsight cost"
- "hindsight constraint"
- "virtual budget" AND RL
- "cost-to-go" AND "relabel"
- "cost-conditioned" AND RL
- "goal-conditioned" AND "safe" AND "cost" AND "hindsight"
- "hindsight" AND "locomotion" AND "constraint"
- "hindsight" AND "biped"
- "hindsight" AND "gait"

**OpenAlex full-text phrase search:**

| Query | Result |
|---|---|
| "hindsight" "cost limit" | 63 results; titles screened, no budget relabeling found |
| "hindsight experience replay" "cost limit" | 9 results; the MDPI items listed in Q3 Gaps were not read |
| "hindsight experience replay" "budget" | 465 results; top 25 screened: CHER, SeqRoute, HALO |
| "cost-to-go" "relabel" | SaFormer, SDT, QPT |
| "hindsight" "safety budget" | TREBI, Imagine-to-Ensure-Safety (not read), risk-bounded navigation |
| "budget relabeling" | SeqRoute, LEMUR (vector rewards), a crowdsourcing paper |
| "cost return" "relabeling" | GAS, RbSL, SaFormer, CDT, SafeMIL (not read), proactive cost generation (not read) |
| "hindsight relabeling" "constrained reinforcement learning" | RbSL, SafeBranch (not read), GCSL with negative feedback (not read) |
| "relabel" "cost threshold" "reinforcement learning" | QPT, DSRL, CDT, SDT |
| "QCP" "hindsight" | Found QCP 2610.37393 |
| "relabel" "budget" "goal-conditioned" | 73 results; top 12 screened, none relevant |
| "sliding window" "cost limit" "reinforcement learning" | 29 results; titles screened, none relevant |
| "token bucket" "hindsight" | 15 results, none about RL budget relabeling |

These OpenAlex queries returned **nothing relevant**:
- "hindsight relabeling" "cost limit" (0 results)
- "hindsight relabeling" "cost budget"
- "constraint relabeling"
- "relabel" "safety budget"
- "token bucket" "safe reinforcement learning"

**Citation trails:** OpenAlex citing works filtered by full text containing "relabel".

| Paper | Citing works | Containing "relabel" |
|---|---|---|
| SeqRoute | 0 | 0 |
| CCPO | 7 | 0 |
| Sauté | 22 | 0 |
| CDT | 18 | 2 (RbSL and a survey) |
| SaFormer | 3 | 0 |

The trails for CHER and Budgeted RL failed with HTTP 429.

**Failed tools:**
- Every Semantic Scholar snippet (full-text) query failed with HTTP 429, so there is no Semantic Scholar full-text coverage.
- Google Scholar was not accessible.

**Web search (paraphrasing engine):**
- Queries: hindsight relabeling cost budget; hindsight "cost limit" goal-conditioned; budget relabeling counterfactual budgets; safe HER Lagrangian CMDP; BCRL; QCP (×3); Sauté/Simmer relabeling; goal-conditioned safe HER cost relabeling; rolling window token bucket legged; hindsight velocity command legged SAC; auto-bidding return-to-go budget relabeling; online relabel safety threshold replay buffer; sliding-window cost CMDP; replenishing or leaky-bucket budget RL; CCAC.
- These surfaced HALO, RADT, CHRL, the low-cost quadruped, TREBI and QCP. None of them returned a rolling-budget relabeling paper.

### Inferences
- A defensible novelty statement, given this search, would be limited to the combination below, with "to our knowledge" wording:
  - relabeling **rolling or refilling (token-bucket or sliding-window) budgets**,
  - with **exact, path-dependent recomputation of bucket levels and violation flags** under each relabeled budget,
  - done **jointly with the commanded velocity (goal)**,
  - inside **online off-policy actor-critic training**,
  - for **several gait budgets at once** (flight, slip, torque saturation).
- The paper should explicitly cite and contrast:
  - SaFormer, for achieved-cost substitution of the cost-limit token in locomotion with a torque cost.
  - SeqRoute, for exact replay of the budget transition with HER framing.
  - HALO, for budget equal to realized cost.
  - CHER, for HER on constraints in an off-policy loop.
  - CDT, GAS and BCRL, for cost-return and budget augmentation.
  - TDM, for joint goal and countdown relabeling.
  - RADT, for joint goal and constraint relabeling.
  - CHRL, for command HER in legged locomotion.
  - LP3, for preference and multiplier relabeling with energy constraints.
  - CRM and LCER, for counterfactual recomputation of an auxiliary tracker.
- Hardware novelty, a low-cost 12-DOF servo biped, can be claimed only once hardware results exist. Every budget-relabeling paper found is simulation-only or non-robotic except QCP (Franka arm) and RbSL (Panda, goal relabeling only).
- These results should not be read as proof of absence. The searches cover arXiv abstracts and OpenAlex full text, but not Semantic Scholar full text, Google Scholar, or the bodies of non-arXiv proceedings. Robotics venues (IROS, ICRA, RA-L) and the operations-research and bidding literature are only partly covered.

### Gaps
- No Semantic Scholar or Google Scholar full-text or citation coverage, because of rate limits and access.
- Title-only screening of large OpenAlex result sets ("hindsight" "cost limit": 63; "relabel" "budget" "goal-conditioned": 73) could miss papers that use different words in their titles.
- Not checked: CCAC (ICLR 2025), Guan et al. 2024, the three MDPI papers listed above, SafeBranch, SafeMIL, the token-bucket RL papers 2511.03279 and 2401.07411, and SceneBot. CCAC and the two MDPI hindsight-plus-safety-budget papers are the highest priority to read before submission.
