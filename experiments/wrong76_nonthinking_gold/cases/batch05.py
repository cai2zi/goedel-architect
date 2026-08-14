from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="cmimc_2025/9",
        steps=(
            Step("$$\n45^5 - 1 = (45 - 1)(45^4 + 45^3 + 45^2 + 45 + 1) = 44 \\cdot P\n$$"),
            Step("$$\nP = 4100625 + 91125 + 2025 + 45 + 1 = 4193821\n$$"),
            Step("After extensive manual checks of divisibility by small primes (up to 2000), and also using modular arithmetic and properties of number theory (e.g., the order of 45 modulo a prime), no divisors are found."),
            Step("This implies that $ 4193821 $ is **prime**."),
            Step("$$\n\\boxed{4193821}\n$$"),
        ),
        nodes=(
            definition("cyclotomicFactor9", "cot_claim", "def cyclotomicFactor9 : ℕ := 4193821", source_steps=(1, 2)),
            claim(
                "factorization_identity9", "lemma factorization_identity9 : (45 : ℕ)^5 - 1 = 44 * cyclotomicFactor9",
                source_steps=(1, 2), label="proved", proof="by norm_num [cyclotomicFactor9]", method="norm_num",
            ),
            claim(
                "nontrivial_factorization9", "lemma nontrivial_factorization9 : cyclotomicFactor9 = 1471 * 2851",
                role="formal_bridge", source_steps=(3, 4), label="proved", proof="by norm_num [cyclotomicFactor9]", method="explicit_factorization",
            ),
            claim(
                "cot_factor_is_prime9", "lemma cot_factor_is_prime9 : Nat.Prime cyclotomicFactor9",
                dependencies=("nontrivial_factorization9",), source_steps=(3, 4), label="disproved",
                proof="by\n  intro hp\n  have hd : 1471 ∣ cyclotomicFactor9 := ⟨2851, by norm_num [cyclotomicFactor9]⟩\n  have h := hp.eq_one_or_self_of_dvd 1471 hd\n  rcases h with h | h <;> norm_num [cyclotomicFactor9] at h",
                method="factors_1471_and_2851",
            ),
            target(
                "gold_cmimc_2025_9", "theorem gold_cmimc_2025_9 : Nat.Prime cyclotomicFactor9 ∧ cyclotomicFactor9 = 4193821",
                dependencies=("factorization_identity9", "cot_factor_is_prime9"), source_steps=(5,), label="blocked_by_dependency",
                statement="The COT reports a composite cyclotomic factor itself as the largest prime factor",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/11",
        steps=(
            Step("$$\n\\text{Linear arrangements} = \\frac{5!}{2! \\cdot 2!} = \\frac{120}{4} = 30\n$$"),
            Step("$$\n30 + 0 + 10 = 40\n$$"),
            Step("$$\n\\text{Number of distinct circular arrangements} = \\frac{40}{10} = 4\n$$"),
            Step("the number of **distinct arrangements equivalent to \"CMIMC\"** is **1** (because it is a single orbit under the group action)."),
            Step("$$\n\\text{Probability} = \\frac{\\text{Number of favorable arrangements}}{\\text{Total number of distinct arrangements}} = \\frac{1}{4}\n$$"),
        ),
        nodes=(
            definition("linearArrangementCount11", "problem_grounding", "def linearArrangementCount11 : ℕ := 30", problem_source_span="five positions with two indistinguishable Cs and two indistinguishable Ms"),
            definition("favorableLinearCount11", "formal_bridge", "def favorableLinearCount11 : ℕ := 5", source_steps=(2, 3, 4), statement="CMIMC has five distinct rotations; reflection produces no new linear word"),
            claim(
                "linear_count11", "lemma linear_count11 : Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2) = linearArrangementCount11",
                source_steps=(1,), label="proved", proof="by norm_num [linearArrangementCount11, Nat.factorial]", method="multiset_permutations",
            ),
            claim(
                "favorable_orbit_size11", "lemma favorable_orbit_size11 : favorableLinearCount11 = 5",
                role="formal_bridge", source_steps=(2, 3, 4), label="proved", proof="by norm_num [favorableLinearCount11]", method="five_rotations_reflection_stabilized",
            ),
            target(
                "gold_cmimc_2025_11", "theorem gold_cmimc_2025_11 : (favorableLinearCount11 : ℚ) / linearArrangementCount11 = 1 / 4",
                dependencies=("linear_count11", "favorable_orbit_size11"), source_steps=(5,), label="disproved",
                proof="by norm_num [favorableLinearCount11, linearArrangementCount11]", method="random_linear_placements_not_uniform_orbits",
                statement="Random bead placements weight linear arrangements, giving 5/30 rather than one of four orbits uniformly",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/13",
        steps=(
            Step("Each round reduces the number of players by **3**, and the game lasts exactly **11 rounds** (since $ 34 - 3 \\times 11 = 1 $)."),
            Step("Therefore, **Michael and James will be in the final round**, and since all players are selected, they will **necessarily battle** in that round."),
            Step("Thus, **Michael and James are guaranteed to be in the same round** (the final one), and **they will battle**."),
            Step("$$\n\\boxed{1}\n$$"),
        ),
        nodes=(
            claim(
                "round_count13", "lemma round_count13 : 34 - 3 * 11 = 1",
                source_steps=(1,), label="proved", proof="by norm_num", method="norm_num",
            ),
            definition(
                "firstRoundSeparatingFailure13", "formal_bridge",
                "def firstRoundSeparatingFailure13 : ℚ := (Nat.choose 32 3 : ℚ) / Nat.choose 34 4 * (3 / 4)",
                source_steps=(2, 3), statement="Michael is chosen without James and loses the first battle",
            ),
            claim(
                "failure_event_positive13", "lemma failure_event_positive13 : 0 < firstRoundSeparatingFailure13",
                role="formal_bridge", source_steps=(2, 3), label="proved",
                proof="by norm_num [firstRoundSeparatingFailure13, Nat.choose]", method="explicit_positive_event",
            ),
            target(
                "gold_cmimc_2025_13", "theorem gold_cmimc_2025_13 : firstRoundSeparatingFailure13 = 0",
                dependencies=("round_count13", "failure_event_positive13"), source_steps=(3, 4), label="disproved",
                proof="by norm_num [firstRoundSeparatingFailure13, Nat.choose]", method="positive_probability_of_elimination_before_meeting",
                statement="A positive-probability first-round event eliminates Michael before he ever battles James",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/14",
        steps=(
            Step("Key observations about the connected components of $ G_{n, k} $:"),
            Step("1 + 3 + 7 + 6 + 22 + 9 + 73 + 12 + 268 + 15 = \\boxed{416}"),
            Step("$$\n\\boxed{416}\n$$"),
        ),
        nodes=(
            definition("rowTotals14", "cot_claim", "def rowTotals14 : List ℕ := [1,3,7,6,22,9,73,12,268,15]", source_steps=(1, 2)),
            definition("cotEvenDiagonalComponentCount14", "cot_claim", "def cotEvenDiagonalComponentCount14 : ℕ := 2", source_steps=(1,)),
            definition("actualG10_10ComponentCount14", "formal_bridge", "def actualG10_10ComponentCount14 : ℕ := 2^9", source_steps=(1,), statement="For k=n=10 every binary string is adjacent only to its complement, giving 2^10/2 complement-pair components"),
            claim(
                "complement_pair_count14", "lemma complement_pair_count14 : actualG10_10ComponentCount14 = 512",
                role="formal_bridge", source_steps=(1,), label="proved", proof="by norm_num [actualG10_10ComponentCount14]", method="count_complement_pairs",
            ),
            claim(
                "cot_even_rule_at_n_equals_k14", "lemma cot_even_rule_at_n_equals_k14 : cotEvenDiagonalComponentCount14 = actualG10_10ComponentCount14",
                dependencies=("complement_pair_count14",), source_steps=(1, 2), label="disproved", proof="by norm_num [cotEvenDiagonalComponentCount14, actualG10_10ComponentCount14]", method="n10_k10_counterexample",
                statement="The table applies the generic even-k value 2 at k=n=10, but G_10,10 has 512 complement-pair components",
            ),
            claim(
                "component_row_sum14", "lemma component_row_sum14 : rowTotals14.sum = 416",
                source_steps=(1, 2), label="proved", proof="by norm_num [rowTotals14]", method="finite_table_sum",
            ),
            target(
                "gold_cmimc_2025_14", "theorem gold_cmimc_2025_14 : rowTotals14.sum = 416",
                dependencies=("cot_even_rule_at_n_equals_k14", "component_row_sum14"), source_steps=(3,), label="blocked_by_dependency",
                statement="The displayed sum is arithmetic-correct for the table, but the table's even diagonal entries are refuted",
            ),
        ),
        fidelity_notes=("The Gold record proves the COT's explicit n-by-n component table aggregation; the parity/span classification is retained as the stated bridge rather than expanded into binary linear algebra.",),
    ),
    Case(
        source_id="cmimc_2025/20",
        steps=(
            Step("- $ S(1) = b_1 = 1 $\n- $ S(2) = b_1 + b_2 + b_3 = 1 + 1 + 2 = 4 $\n- $ S(3) = b_1 + \\cdots + b_9 = 8 $\n- $ S(4) = 14 $"),
            Step("$$\nS(1) = 4 = 2^2 \\\\\nS(2) = 8 = 2^3 \\\\\nS(3) = 14 = 2^4 - 2 \\\\\nS(4) = 26 = 2^5 - 2\n$$"),
            Step("$$\nS(m) = 2^{m+1} - 2 \\quad \\text{for } m \\geq 2\n$$"),
            Step("$$\n\\sum_{n=1}^{3^{2025}} b_n = S(2025) = 2^{2026} - 2\n$$"),
            Step("$$\n\\boxed{2^{2026} - 2}\n$$"),
        ),
        nodes=(
            definition("firstS1_20", "cot_claim", "def firstS1_20 : ℕ := 1", source_steps=(1,)),
            definition("secondS1_20", "cot_claim", "def secondS1_20 : ℕ := 4", source_steps=(2,)),
            claim(
                "cot_base_values_consistent20", "lemma cot_base_values_consistent20 : firstS1_20 = secondS1_20",
                source_steps=(1, 2), label="disproved", proof="by norm_num [firstS1_20, secondS1_20]",
                method="same_symbol_s1_given_two_values",
            ),
            target(
                "gold_cmimc_2025_20", "theorem gold_cmimc_2025_20 : firstS1_20 = secondS1_20 ∧ (2 : ℕ)^2026 - 2 = 2^2026 - 2",
                dependencies=("cot_base_values_consistent20",), source_steps=(3, 4, 5), label="blocked_by_dependency",
                statement="The extrapolated pattern starts from mutually inconsistent displayed base values",
            ),
        ),
    ),
)
