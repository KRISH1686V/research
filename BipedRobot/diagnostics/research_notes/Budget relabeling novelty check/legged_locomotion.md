# Budget relabeling novelty check: legged locomotion prior work

Claim tested: for a low-cost 12-DOF hobby-servo biped, limits on flight, stance slip and torque saturation are adjustable budgets inside the goal, relabeled in hindsight with the velocity command during online off-policy training (SAC + HER, mirror augmentation), so one run covers many gait limits with no gait rewards.

Search date: 2026-11-04.

## Which legged-robot papers come closest, and does any break the claim?

### Takeaway
No paper I read breaks the full claim, but two sub-claims are already taken. QuietPaw gives a legged policy a cost threshold as input and changes it after training, and CHRL relabels velocity commands in an off-policy legged replay buffer. I found no legged paper that relabels a constraint limit in hindsight, which is the part of the claim still standing. This is a "not found" result, not proof of absence.

### Cited Findings
How these were read: quotes marked "raw" were pulled directly from the paper's HTML or PDF text. Other findings come from a fetch tool that answers questions over the paper's text, so check them by hand before citing them in the journal paper.

Closest papers, element by element:

| Paper | Robot | Limit given to policy? | Changed after training? | Hindsight relabeled | Algorithm | Gait reward? |
|---|---|---|---|---|---|---|
| [QuietPaw](https://arxiv.org/html/2503.05035v1) ([IROS 2025 listing](https://www.nature.com/nature-index/article/10.1109/iros60139.2025.11246267)) | Unitree Go2, real | Yes: one noise cost threshold | Yes, real time | Nothing | PPO-Lagrangian | Yes, then decayed |
| [CHRL](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564515/) | 15 kg quadruped, 12 actuated joints, real | No | No | Velocity and yaw command only | REDQ (teacher), DAgger (student) | No air-time reward mentioned; foot slip is a curriculum-ramped penalty, not a budget |
| [GentleHumanoid](https://arxiv.org/html/2511.04679v1) | Unitree G1, real | Yes: force limit, as an observation | Yes | Nothing | PPO | Not checked; the limit concerns upper-body contact, not gait |
| [Bolt with CaT](https://arxiv.org/html/2508.02194) | Low-cost biped, 6 torque-controlled DoF, point feet | No | No; gait change needs retraining | Nothing | PPO + CaT | No; gait is enforced by fixed constraints |
| [LoComposition](https://arxiv.org/html/2606.15896) | Go2 real; Spot and ANYmal C in simulation | No; fixed torque and other limits | Not described | Nothing | PPO + CaT | None in the final policy |
| [Hindsight Preference Replay](https://arxiv.org/html/2601.11604v1) | MO-Gymnasium simulation only, including walker2d and humanoid | Reward weights, not limits | Not checked | Preference (reward-weight) vector | CAPQL, which extends SAC | Benchmark rewards |
| [CCPO](https://arxiv.org/pdf/2310.03718) | Bullet-Safety-Gym simulation: Ball, Car, Drone, Ant | Yes: threshold | Yes, zero-shot | No relabeling in the parts read; value estimation at unseen thresholds | Off-policy baselines V-SAC-Lag, V-DDPG-Lag | Not applicable |
| [PROMO](https://arxiv.org/html/2610.01260v1) | Unitree Go2 | Reward weights, not limits | Yes | Nothing | PPO | Yes: fixed contact-timing prior |
| [Masuda and Takahashi](https://arxiv.org/html/2204.03897v4) | ROBOTIS-OP3, real, Dynamixel | No | No | Not mentioned | SAC | Yes: phase-synchronized gait term |

Most important details:
- QuietPaw, raw: "we extend the Safe RL formulation to a constraint-conditioned RL setting, where the policy is conditioned on a distribution of noise thresholds" — [QuietPaw](https://arxiv.org/html/2503.05035v1)
- QuietPaw, raw: "During deployment, the ϵ acts as a noise reduction effect steering command, allowing real-time adjustment of the noise minimization." — [QuietPaw](https://arxiv.org/html/2503.05035v1)
- CHRL's relabeling samples a new command near the original (linear velocity within ±0.3 m/s, yaw within ±0.5 rad), limited to the current curriculum range, and recomputes the reward from extra robot state stored in each transition. It does not use the achieved velocity as the goal. — [CHRL](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564515/)
- Bolt paper, raw: "Switching between walking and jumping behaviors requires retraining the network due to the mutually exclusive nature of the constraints." — [Bolt with CaT](https://arxiv.org/html/2508.02194)
- LoComposition's heading, raw: "Constraint-based gait priors harm terrain traversal and prevent efficient gaits". Its text also says encoding air-time and contact-count priors as constraints "is detrimental in our formulation". — [LoComposition](https://arxiv.org/html/2606.15896)

### Inferences
- A claim such as "first legged policy conditioned on constraint limits" would be false because of QuietPaw. "First velocity-command HER on legged robots" would be false because of CHRL.
- A defensible wording, based only on what I searched: to our knowledge, the first legged-locomotion work to relabel constraint budgets in hindsight, and the first to condition one policy on flight, slip and torque budgets together with no gait reward. Say "to our knowledge" and name the search scope.
- These papers should be cited and contrasted directly: QuietPaw, CHRL, GentleHumanoid, Bolt with CaT, LoComposition and Hindsight Preference Replay.
- A reviewer could argue budget relabeling is a small step from CHRL plus QuietPaw. CHRL already recomputes rewards from stored state, including a foot-slip term, and QuietPaw already conditions on a cost threshold. The paper should show what relabeling budgets adds over sampling them per environment, as QuietPaw does.
- Most of these papers show real hardware. Project memory says the biped's hardware is on hold, so a simulation-only result supports the "hobby-servo" element only through the robot model. Reviewers may press on this.

### Gaps
- I cannot rule out a paper that mentions relabeling or limit inputs only in its body text. The arXiv searches covered abstracts only.
- Very recent preprints may not be indexed yet. Several close papers date from September and October 2026.

## Does any legged-robot paper give limits, budgets or thresholds to the policy as inputs, or change them after training without retraining?

### Takeaway
Yes on both. QuietPaw is the direct precedent, with a true constrained formulation on a real quadruped. GentleHumanoid is a second case, using a soft penalty rather than a constraint. Neither covers flight, slip or torque limits, and neither relabels anything.

### Cited Findings
Limit given to the policy:
- QuietPaw solves J_c ≤ ϵ for all ϵ in a set. Raw: "each constraint level ϵ i is uniformly interpolated over [ 0 , 1 ]", and each environment is assigned one level, with its own Lagrange multiplier. — [QuietPaw](https://arxiv.org/html/2503.05035v1)
- QuietPaw's single cost combines contact force and impact velocity. No torque, slip or flight constraint was found in the text read. — [QuietPaw](https://arxiv.org/html/2503.05035v1)
- Warning for the claim. QuietPaw, raw: "Simply concatenating the noise constraint into the policy input, often fails to induce meaningful behavioral differences". It uses a successor-feature critic instead and treats plain concatenation as a baseline. — [QuietPaw](https://arxiv.org/html/2503.05035v1)
- GentleHumanoid lists τsafe as an observation, "that can be changed by use during deployment". It is resampled every 5 seconds within 5 to 15 N during training. — [GentleHumanoid](https://arxiv.org/html/2511.04679v1)
- GentleHumanoid enforces the limit with a clamp on a virtual spring force and an indicator reward penalty, not a Lagrangian or constrained-MDP cost. — [GentleHumanoid](https://arxiv.org/html/2511.04679v1)
- CCPO conditions on thresholds with zero-shot adaptation, but its experiments are simulated Run and Circle tasks with Ball, Car, Drone and Ant. Its baselines put the threshold directly in the state of off-policy SAC-Lag and DDPG-Lag. — [CCPO](https://arxiv.org/pdf/2310.03718)

Limit changed after training by other means:
- The hexapod whole-body follower tunes deployment constraints without retraining the policy, but the limits live in an optimization layer and are not policy inputs. — [Whole-Body Constrained Learning](https://arxiv.org/html/2506.05115)
- SAC-Loco (here SAC means "Safe and Adjustable Compliant", not Soft Actor-Critic) takes a compliance level as a command and has a tunable switching threshold ε applied outside the network. It uses PPO on a Go2. — [SAC-Loco](https://arxiv.org/html/2509.23223v1)

Preference weights rather than limits:
- PROMO: a preference vector over tracking, stability and efficiency, changeable at runtime; PPO on Go2. — [PROMO](https://arxiv.org/html/2610.01260v1)
- Leng et al.: a weight vector between tracking and force compliance, adapted online; PPO on Booster T1. — [Preference-Conditioned Multi-Objective RL](https://arxiv.org/html/2510.10851v1)
- PCHC: weights sampled from a Dirichlet distribution and concatenated to the observation; PPO. — [PCHC](https://arxiv.org/html/2603.24047v1)
- "One Policy, Many Styles" reports that early aggregation gives a wider behavior range but more unintended changes. Abstract only. — [One Policy, Many Styles](https://arxiv.org/abs/2610.28925)

Limits that stay out of the policy:
- SATA grows the torque limit on a schedule during training. The observation holds actual torque, not the limit, and deployment uses full capacity. — [SATA](https://arxiv.org/html/2502.12674v1)
- ACPO adapts the cost budget during training toward a fixed desired budget (2 for each cost in the simulated quadruped task). I saw no conditioning on the budget in the text searched. — [ACPO](https://arxiv.org/pdf/2410.20786)
- ECO on the BRUCE humanoid searches energy thresholds of 20 to 100 J by training at each one. Raw: "Each threshold search takes around 4 hours of wall-clock time". — [ECO](https://arxiv.org/html/2602.06445)
- FAITH: the safety margin is not listed as a policy input. — [FAITH](https://arxiv.org/html/2610.12432v1)
- Digit safe-RL paper: fixed ALIP-based bounds; PPO. — [Model-Informed Safe RL](https://arxiv.org/html/2609.34486v1)
- APEX: no torque or power limit in the policy inputs. — [APEX](https://arxiv.org/html/2602.11143v3)

### Inferences
- QuietPaw's concatenation warning is the likeliest reviewer challenge. The claim's results should show that behavior actually changes with each budget, not only that the budget is in the observation. QuietPaw's finding comes from one on-policy setup, so it may not carry over to off-policy training with relabeling.
- "Budget" means different things across these papers. QuietPaw uses a Lagrangian constraint and GentleHumanoid an indicator penalty, so the paper should state its enforcement mechanism exactly.
- ECO shows the cost the claim aims to remove, since each threshold needs its own training. It makes a useful motivating citation.

### Gaps
- QuietPaw does not describe exactly how ϵ enters its actor network in the text read. The conditioning itself is explicit in its equations, written as π(·|s, ϵ).
- QuietPaw's citers ("Quiet Humanoid Walking With NMPC-Informed RL", "Contact Force Estimation-Guided RL") were listed by [OpenAlex](https://api.openalex.org/works?search=QuietPaw) but not read.
- Queries run:
  - "constraint-conditioned policy legged locomotion cost limit as input": no conditioned legged paper in the results.
  - "adjustable constraint thresholds quadruped policy without retraining": surfaced CCPO (simulation only).
  - "policy conditioned on torque limit legged robot": no paper with the torque limit as input.
  - "safety-conditioned humanoid policy threshold input": led to GentleHumanoid through a [survey](https://arxiv.org/html/2609.13236v1).
  - "conditional constraint threshold policy legged PPO-Lagrangian thresholds sampled per environment humanoid": no match.
  - "contact force limit or energy limit given as command to legged locomotion policy": no match.
- arXiv abstract searches run:
  - Constraint-, threshold-, budget-, limit-, cost- and safety-conditioned with legged terms: 6 results, none a limit-conditioned policy on titles and the FAITH read. This query did not surface QuietPaw, so it misses relevant papers.
  - Constraint threshold, cost limit, cost budget or safety budget with legged terms: 8 results, with ROGER and ACPO relevant but neither conditioned.
  - Torque limit, power limit or energy budget with "conditioned": 3 results; APEX read, the other two judged on title only.

## Does any legged-robot paper use hindsight relabeling, and what exactly is relabeled?

### Takeaway
CHRL is the only legged paper I found that relabels inside an off-policy replay buffer, and it relabels commands only. Humanoid papers do relabel achieved goals, but in supervised or planner datasets, not in RL replay. I found nothing that relabels environment parameters, reward weights or constraints on a legged robot, and nothing named HER-Loco.

### Cited Findings
- CHRL's HER section describes commands only: "We sample new command cnewcmd by adding neighboring regions to the sampling distribution". It does not mention relabeling terrain, reward weights or constraints. Five million simulated steps from four agents in PyBullet, then zero-shot deployment. — [CHRL](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564515/)
- CHRL's curriculum ramps its reward coefficients, and the paper does not say whether relabeled rewards use current or stored coefficients. — [CHRL](https://pmc.ncbi.nlm.nih.gov/articles/PMC11564515/)
- Humanoid-DART on G1, raw: "the pipeline relabels physically valid rollouts with their achieved task metrics". This feeds a diffusion planner's elite set; the tracker uses PPO. — [Humanoid-DART](https://arxiv.org/html/2606.26855)
- Distance-conditioned transport on G1 relabels the goal of each retained trajectory by its achieved object placement, then distills by behavior cloning and fine-tunes with PPO. — [Distance-Conditioned Object Transport](https://arxiv.org/html/2609.21467v1)
- Hindsight Preference Replay, raw: "retroactively relabels stored transitions with alternative preferences". Off-policy and SAC-derived, in simulation only, with no constraints. Tasks include mo-walker2d-v5, described as "A bipedal walker". — [Hindsight Preference Replay](https://arxiv.org/html/2601.11604v1)
- Papers that use the word but do not relabel RL goals:
  - SWAP places obstacles in space the motion did not sweep ("hindsight labeling"); PPO low-level controller. — [SWAP](https://arxiv.org/html/2610.25711v1)
  - SceneBot builds terrain and objects to fit a motion ("hindsight scene reconstruction"). — [SceneBot](https://arxiv.org/html/2606.27581v1)
  - TACOS generates contact-based labels for footstep planning. — [TACOS](https://arxiv.org/html/2610.37806v1)
  - The Rhoban team paper uses "hindsight" in its everyday sense. — [Rhoban](https://arxiv.org/html/2402.00612v1)
- A keyword search of the paper text for relabel, hindsight and HER found nothing in these:
  - [FastTD3](https://arxiv.org/html/2505.22642)
  - [FastSAC](https://arxiv.org/html/2512.01996)
  - [FlashSAC](https://arxiv.org/html/2604.04539v2)
  - [WarpSAC](https://arxiv.org/html/2608.24479)
  - [Pretraining and Finetuning](https://arxiv.org/html/2601.21363v3)
  - [SMPC Offline-to-Online](https://arxiv.org/html/2608.12063v2)
  - [Stacked Payloads MORL](https://arxiv.org/html/2610.05819v1)
- Outside legged robots: quality-controllable policies relabel successful rollouts by realized score, on manipulation only. — [Iterative Hindsight Relabeling](https://arxiv.org/html/2610.37393v1)

### Inferences
- On a biped, the answer depends on the wording. Hindsight relabeling on a humanoid exists (Humanoid-DART and the transport paper). Relabeling velocity commands or limits in an off-policy replay buffer on a biped was not found.
- Hindsight Preference Replay is the closest method match: a conditioning vector relabeled in a SAC-style buffer on simulated locomotion, including a simulated biped. The conditioning vector is reward weights, not limits. A reviewer is likely to find it, so it should be cited.

### Gaps
- CHRL citation trail:
  - [OpenAlex](https://api.openalex.org/works?filter=cites:W4404355355) lists 15 citers. None of the abstracts mention hindsight, relabeling, replay or off-policy methods.
  - Most full texts were not read, including "Curriculum-Based Deep RL for Robust Biped Push Recovery" (AIM 2026) and "TSIE" (biped). A citer could use relabeling without saying so in its abstract.
  - Semantic Scholar returned rate-limit errors (HTTP 429) for CHRL, so arXiv-only citers may be missing.
- Queries run:
  - "HER-Loco hindsight experience replay legged locomotion": no paper with that name.
  - "hindsight relabeling biped humanoid locomotion reinforcement learning": only CHRL as a locomotion match.
  - "quadruped goal relabeling off-policy SAC velocity command hindsight real robot": only CHRL.
  - "hindsight relabeling reward weights conditioned locomotion policy": no legged match.
  - "relabel commands achieved velocity replay buffer legged robot": only CHRL.
  - "HER goal relabeling velocity tracking SAC humanoid MuJoCo Playground": surfaced Humanoid-DART, with no SAC plus HER locomotion paper.
  - "goal-conditioned hindsight experience replay quadruped robot locomotion sparse reward gait parameters relabeled": no match.
- arXiv abstract searches run:
  - "hindsight" with legged terms: 8 results.
  - "relabeling" and "relabel" with legged terms: 6 each.
  - "hindsight" with locomotion, walking or gait: 12 results.
  - "experience replay" with locomotion and goal or command: 0 results.
  - Only the papers listed above were relevant.

## What do the named papers actually do?

### Takeaway
None of the named papers feeds a limit to the policy or relabels anything. Several already show parts of the claim separately: constraints replacing gait rewards, behavior conditioning, and off-policy humanoid training with symmetry augmentation.

### Cited Findings
| Paper | Limit or conditioning | After training | Algorithm, robot | Gait or relevant terms |
|---|---|---|---|---|
| [CaT](https://arxiv.org/html/2403.18765) | Fixed limits in a hyperparameter table; not in the listed state | Not described | PPO, Solo-12 | Per-foot air time (0.25 s minimum), contact count (2) and torque as constraints |
| [Not Only Rewards](https://arxiv.org/html/2308.12517) | 11 fixed constraints; thresholds raised adaptively in training only; not in the listed observations | Experiments train new policies | Modified IPO with TRPO; Raibo and Mini-cheetah real, more in simulation | No gait reward; phase-based gait-pattern constraint; slip through a contact-velocity constraint; no flight constraint |
| [Whole-Body Constrained Learning](https://arxiv.org/html/2506.05115) | Limits in the optimization follower, not policy inputs | Deployment constraints tunable | PPO, 18-DoF hexapod | No gait-pattern reward; slip handled by a deployment constraint; no flight constraint |
| [Walk These Ways](https://ar5iv.labs.arxiv.org/html/2212.03238) | 8 behavior parameters as inputs, tracked targets rather than limits | Operator tunes in real time | PPO, Unitree Go1 | Auxiliary and contact-schedule rewards are required |
| [Multi-AMP](https://ar5iv.labs.arxiv.org/html/2203.14912) | One-hot style selector | Discrete switching | PPO in its algorithm box; wheeled-legged quadruped | Discriminator style reward |
| [Gait-conditioned humanoid](https://arxiv.org/html/2505.20619) | One-hot gait ID | Gait selectable | PPO with LSTM, Unitree G1 | Per-gait contact-pattern rewards |
| [Barrier-based style rewards](https://arxiv.org/html/2409.15780) | Bounds fixed per mode; not in the observation | Not described; tuned before training | PPO, HOUND and HOUND2 | Relaxed log barriers on a gait cycle and clearance |
| [ROGER](https://arxiv.org/abs/2510.10759) | Reward gains adapted online during learning | Not described | Parameter-space learning; 60 kg quadruped | Near-zero violation reported; abstract and a [text search](https://arxiv.org/html/2510.10759) only |
| [Lipschitz-constrained policies](https://arxiv.org/html/2410.11825) | Lipschitz bound folded into a fixed weight; not an input | Not described | PPO; GR1T1, GR1T2, H1, Berkeley Humanoid | Gait style rewards kept |
| [FastTD3](https://arxiv.org/html/2505.22642) | None | Not applicable | FastTD3, Booster T1 real | Reward tuned with stronger penalties |
| [FastSAC](https://arxiv.org/html/2512.01996) | None | Not applicable | FastSAC, G1 and T1 | Foot-height tracking and other terms; symmetry augmentation used |

- FastTD3, raw: "the first documented instance of a successful deployment of an off-policy RL policy on a full-size humanoid robot in the real world". This is limited to full-size humanoids. — [FastTD3](https://arxiv.org/html/2505.22642)
- Reward-Adaptive RL for bipeds uses dynamic reward weights with a multi-head critic. Search snippet only; those results did not describe the weights as a policy input. — [HDPG](https://arxiv.org/pdf/2107.01908)
- Note on the CaT air-time constraint: it sets a minimum air time per foot, which is not the same as a cap on both feet being off the ground. — [CaT](https://arxiv.org/html/2403.18765)

### Inferences
- Walk These Ways is the classic "one policy, many behaviors" citation, but its parameters are reward-tracked targets. The claim's contrast is limits versus targets, and no gait rewards.
- SAC with symmetry augmentation on a humanoid is already in FastSAC, so mirror augmentation adds no novelty by itself.

### Gaps
- Only the abstract and keyword hits were read for ROGER.
- The Multi-AMP arXiv ID was not supplied. I used [2203.14912](https://arxiv.org/abs/2203.14912) after 2209.13362 turned out to be an unrelated paper.
- The last 36,000 characters of Not Only Rewards were not read. They are probably references and appendix.

## What off-policy or SAC locomotion exists on real low-cost or servo-driven bipeds and small humanoids?

### Takeaway
Off-policy training on real small humanoids exists, but on ROBOTIS OP3 with Dynamixel servos and with no relabeling or budgets. Every other low-cost or hobby-servo biped I could check uses PPO or names no algorithm. I found no SAC or other off-policy walking on a Feetech-class hobby-servo biped.

### Cited Findings
Off-policy on real bipeds:
- Haarnoja et al. use MPO with a distributional critic ("an off-policy actor-critic RL algorithm") on OP3, "a low-cost battery-powered, miniature humanoid platform", with 20 Dynamixel XM430-350-R servos and zero-shot transfer. The text describes no dedicated gait reward. It has a fixed penalty on torque peaks above 5 Nm. — [Soccer](https://ar5iv.labs.arxiv.org/html/2304.13653)
- Masuda and Takahashi state "we use Soft-Actor-Critic" on OP3 (20 DoF, Dynamixel XM430-W350), with a phase input and a gait term (1.0 s cycle, 0.7:0.3 support to swing). — [OP3 sim-to-real](https://arxiv.org/html/2204.03897v4)
- SAC pretraining and real finetuning on Booster T1; the reward includes a swing-leg reference. — [Pretraining and Finetuning](https://arxiv.org/html/2601.21363v3)
- FlashSAC on a 29-DoF G1, with reward terms including foot-slip and torque penalties. — [FlashSAC](https://arxiv.org/html/2604.04539v1)

Low-cost and servo bipeds, none off-policy:
- ToddlerBot: PPO, Dynamixel servos, 30 DoF, under $6,000, ZMP reference plus feet air time. — [ToddlerBot](https://arxiv.org/html/2502.00893)
- Berkeley Humanoid Lite: PPO, 3D-printed cycloidal actuators, under $5,000. — [Berkeley Humanoid Lite](https://arxiv.org/html/2504.17249)
- Duke Humanoid: Motorevo brushless motors, 10 DoF, reward table with feet air time and contact pattern. PPO does not appear in the text; the algorithm is not named. — [Duke Humanoid](https://arxiv.org/html/2409.19795)
- Open Duck Mini: BOM under $400 ([README](https://github.com/apirrone/Open_Duck_Mini)). PPO through Brax ([runner code](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/common/runner.py)), and `USE_IMITATION_REWARD = True` ([joystick code](https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/main/playground/open_duck_mini_v2/joystick.py)). Servo brand comes from secondary sources only: Feetech ST3215 ([DEV](https://dev.to/frankfu/understanding-reinforcement-learning-through-openduck-1if0)) and "hobby servos" ([Hackaday](https://hackaday.com/2025/04/05/disneys-bipedal-bdx-series-droid-gets-the-diy-treatment)).
- Microduck: PPO on mjlab, Dynamixel XL330, 14 servos per its [repo](https://github.com/pollen-robotics/microduck_rl). Conflicts: [another listing](https://gittrend.io/repo/pollen-robotics/microduck) says fifteen servos, and a [news piece](https://www.techtimes.com/articles/329379/20261022/sim-real-gap-narrowed-399-duck-robot-isaac-lab-30-proves-pure-simulation-training-works.htm) says Isaac Lab.
- Bimo: 8 STS-3215 servos, RL forward walking reported sim-to-real, algorithm not named. — [Bimo](https://discourse.openrobotics.org/t/bimo-an-open-source-bipedal-robotics-kit-with-sim-to-real-transfer/57846)
- EVAL-03: PPO, gear-driven, 12 active DoF, 1.73 kg, symmetric augmentation, air-time and contact rewards. — [EVAL-03](https://arxiv.org/html/2504.00614)
- OP3 tele-loco-manipulation: custom gait reward, algorithm not named. — [Miniature Humanoid](https://arxiv.org/html/2607.20399)

Low-cost quadrupeds and real-world off-policy:
- Mini Pupper 2 (about $300, brushed position servos): PPO, with a foot-slip penalty. — [Cost-Constrained Quadrupedal Hardware](https://arxiv.org/html/2607.26434v2)
- De Jong et al. apply SAC on low-cost Pupper and SpotMicroAI platforms, but through a Bezier gait generator with policy residuals, so the gait is not learned from scratch. The SAC findings I read were from simulation. — [Learning to walk with SAC](https://staff.fnwi.uva.nl/a.visser/publications/LearningToWalk.pdf)
- Real-world quadruped learning, abstract only:
  - [A Walk in the Park](https://arxiv.org/abs/2208.07860): 20 minutes on a real quadruped.
  - [Ha et al.](https://arxiv.org/abs/2002.08550): safety-constrained RL on Minitaur.
  - [Towards General and Autonomous Learning](https://arxiv.org/abs/2008.12228): off-policy multi-task learning.

### Inferences
- "Off-policy on a low-cost small humanoid" is not new because of OP3. The narrower gap is off-policy training on a true hobby-servo biped, and that gap is thinner than it sounds: OP3 is described as low-cost, though it uses Dynamixel servos.
- Exact 12-DOF hobby-servo wording will probably hold up, but it is a platform distinction, not a method contribution.

### Gaps
- No academic paper was found pairing Feetech or STS3215 servos with RL biped walking. Hobby projects are poorly indexed, so repos may exist.
- The last 36,000 characters of the soccer paper were not read, so "no dedicated gait reward" rests on the first 100,000 characters.
- Queries run:
  - "off-policy SAC real low-cost bipedal robot servo sim-to-real walking": no SAC on a budget servo biped.
  - "Feetech servo humanoid reinforcement learning locomotion low-cost biped": no academic match.
  - "soft actor-critic bipedal humanoid hardware walking Dynamixel real robot SAC gait": no new real biped beyond OP3.
  - "Open Duck Mini Feetech STS3215 servos": secondary sources only.
  - "Microduck Pollen Robotics reinforcement learning Isaac Lab servo PPO rewards": PPO confirmed; reward terms not found.
- arXiv abstract searches run:
  - SAC with biped or humanoid and real hardware: 6 results.
  - Servo, Feetech, STS3215 or Dynamixel with RL and biped or humanoid: 4 results.
  - Off-policy with humanoid or biped and hardware: 9 results.
  - None was a hobby-servo biped. These were judged on titles, except those listed above.
- Reward terms for Microduck and Bimo could not be confirmed.

## Does any paper train without any gait reward, using a budget or constraint on flight or slip alone?

### Takeaway
I found none. Training with no gait reward is well established, and fixed contact or slip constraints exist, but always with other gait structure or a different objective shaping the gait. No paper uses adjustable flight or slip budgets as the only gait-shaping signal.

### Cited Findings
- LoComposition's final policy has only velocity tracking, a mechanical-power penalty and fixed operational limits. Raw: it "removes explicit gait priors (including air-time, contact-count, and foot-clearance targets)". Slip and flight are not among its limits; energy shapes the gait. — [LoComposition](https://arxiv.org/html/2606.15896)
- Bolt with CaT has no gait reward, but its fixed gait constraints include |n_contact − 1| and foot air time. Slip is not a training constraint. — [Bolt with CaT](https://arxiv.org/html/2508.02194)
- Not Only Rewards uses 3 reward terms and no gait reward, but it constrains a phase-based gait pattern. — [Not Only Rewards](https://arxiv.org/html/2308.12517)
- Fu et al. use velocity, energy and survival terms on an A1 with PPO, "without relying on demonstrations or predefined motion heuristics". — [Fu et al.](https://ar5iv.labs.arxiv.org/html/2111.01674)
- SATA has no explicit gait reward, and its authors note the policy produces only a basic walking gait. — [SATA](https://arxiv.org/html/2502.12674v1)
- Koseki et al. use SAC with no reference motion on a simulated sagittal biped and get walking and running from a speed command and an energy-variation penalty. They add leg-swing and symmetry terms, and hardware is not addressed. — [Koseki et al.](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.1054239/full)
- QuietPaw, raw: "We decay the gait-related reward using a sigmoid decay function at the 2k step". So it uses a gait reward first and then reduces it. — [QuietPaw](https://arxiv.org/html/2503.05035v1)

### Inferences
- "No gait rewards" is not a novelty on its own. Its value in the claim comes from pairing it with adjustable budgets.
- Bolt's |n_contact − 1| constraint is the nearest thing to a flight limit on a biped. It is fixed, and it also forbids double support, so it is stricter than a both-feet-off budget.
- LoComposition's finding that air-time and contact-count constraints are harmful is a counterpoint a reviewer may raise. It concerns target-style priors on quadrupeds over terrain, not flight budgets on a biped, so it does not settle the question.
- Energy appears to do the gait shaping in several of these papers. If the claim's policy walks with only flight, slip and torque budgets, that would be a distinct result. It would also need checking for degenerate gaits such as shuffling or tapping, because Fu et al. report that policies without the energy term tapped at high frequency and did not transfer ([search summary of the paper](https://arxiv.org/pdf/2111.01674); I did not confirm this in the text).

### Gaps
- ECO's and the [smoothness-constraint paper](https://arxiv.org/html/2609.24552)'s reward tables were not fully read, so their gait-reward status is unverified.
- Queries run:
  - "no gait reward constraint on flight phase bipedal walking emergence reinforcement learning": no match.
  - "emergent gait without gait reward biped hobby servo foot slip penalty only reinforcement learning": no match.
  - "adaptive reward weights bipedal locomotion policy input reward-adaptive reinforcement learning": no weights-as-input paper.
- arXiv abstract searches run:
  - Flight phase, flight time, air time, foot slip or foot slippage with "constraint", RL and legged terms: 1 result (the hexapod paper).
  - "Without gait", "no gait", "gait-free" and similar with RL: 255 total, of which only the newest 30 titles were seen, so this search is weak.
- Citation trails, which limit every "not found" in these notes:
  - CaT: Semantic Scholar failed (HTTP 429). OpenAlex lists only 23 citers, which looks like an undercount, and none passed the keyword filter. This is the weakest trail and is worth checking by hand in Google Scholar.
  - Not Only Rewards: 126 citers on Semantic Scholar and 73 on OpenAlex, filtered by title and abstract keywords. This filter is how QuietPaw and LoComposition surfaced.
  - Walk These Ways: 62 citers on OpenAlex, filtered the same way, with no relevant hit.
