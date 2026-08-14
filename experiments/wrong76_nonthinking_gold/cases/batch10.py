from __future__ import annotations

from gold_dsl import Case, Step, claim, definition, target


CASES = (
    Case(
        source_id="cmimc_2025/15",
        steps=(
            Step("a \"double\" is defined by **value**, not by identity."),
            Step("Each removal removes a prefix of the deck, and the process restarts from the beginning of the remaining cards."),
            Step("$$\nE[N] = \\sum_{k=1}^\\infty P(N \\geq k)\n$$"),
            Step("$$\nE(n) = 1 + \\mathbb{E}[E(n - k)]\n$$"),
            Step("$$\n\\boxed{3}\n$$"),
        ),
        nodes=(
            definition("distinctTwoRemovalCount15", "formal_bridge", "def distinctTwoRemovalCount15 : ℕ := 0", source_steps=(1, 2, 3, 4), statement="A two-card deck with distinct values contains no double and causes no removal"),
            claim("distinct_two_base15", "lemma distinct_two_base15 : distinctTwoRemovalCount15 = 0", role="formal_bridge", source_steps=(1, 2), label="proved", proof="by rfl", method="scan_contains_no_double"),
            claim(
                "cot_recurrence_unconditional_one15", "lemma cot_recurrence_unconditional_one15 : distinctTwoRemovalCount15 = 1",
                dependencies=("distinct_two_base15",), source_steps=(3, 4), label="disproved",
                proof="by norm_num [distinctTwoRemovalCount15]", method="no_double_branch_has_zero_removals",
                statement="The displayed recurrence adds one even on permutations containing no double",
            ),
            target("gold_cmimc_2025_15", "theorem gold_cmimc_2025_15 : distinctTwoRemovalCount15 = 1 ∧ (3 : ℚ) = 3", dependencies=("cot_recurrence_unconditional_one15",), source_steps=(5,), label="blocked_by_dependency", statement="The stated expectation three rests on an invalid unconditional recurrence"),
        ),
    ),
    Case(
        source_id="cmimc_2025/18",
        steps=(
            Step("A regular polygon with an **even number of sides** can be divided into **non-overlapping parallelograms**."),
            Step("A regular polygon can be **divided into non-rectangular parallelograms** using lines that are **not perpendicular**."),
            Step("Therefore, **it is possible to tile the polygon using only non-rectangular parallelograms**, resulting in **zero rectangles**."),
            Step("$$\n\\boxed{0}\n$$"),
        ),
        nodes=(
            definition("generatorDirectionCount18", "problem_grounding", "def generatorDirectionCount18 : ℕ := 8960 / 2", problem_source_span="a centrally symmetric regular 8960-gon has 4480 unoriented edge directions"),
            definition("perpendicularDirectionPairs18", "formal_bridge", "def perpendicularDirectionPairs18 : ℕ := generatorDirectionCount18 / 2", source_steps=(1, 2, 3), statement="Opposite-direction generators separated by a quarter turn form unavoidable rectangular tiles in a zonogon tiling"),
            claim("unavoidable_rectangles18", "lemma unavoidable_rectangles18 : perpendicularDirectionPairs18 = 2240", role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by norm_num [perpendicularDirectionPairs18, generatorDirectionCount18]", method="count_quarter_turn_direction_pairs"),
            target(
                "gold_cmimc_2025_18", "theorem gold_cmimc_2025_18 : perpendicularDirectionPairs18 = 0",
                dependencies=("unavoidable_rectangles18",), source_steps=(3, 4), label="disproved",
                proof="by norm_num [perpendicularDirectionPairs18, generatorDirectionCount18]", method="zonogon_pair_tiles_include_perpendicular_pairs",
                statement="Every parallelogram tiling of the regular zonogon has a tile for each generator pair, including 2240 perpendicular pairs",
            ),
        ),
    ),
    Case(
        source_id="cmimc_2025/32",
        steps=(
            Step("The algorithm **terminates with probability 1** when 7 is written down."),
            Step("$$\np(7) = 1\n$$"),
            Step("This is equivalent to computing the **expected number of distinct numbers (excluding 7) written down before 7 is written down**, since each number is written down at most once (due to the stopping condition)."),
            Step("$$\np(1) + p(2) + p(3) + p(5) + p(6) = \\frac{5}{7}\n$$"),
            Step("$$\n\\boxed{\\frac{5}{7}}\n$$"),
        ),
        nodes=(
            definition("rootTrace32", "formal_bridge", "def rootTrace32 : List ℕ := [4,4]", source_steps=(1, 2, 3), statement="Choose an edge not incident to the root; the root label 4 is written in two consecutive turns"),
            claim(
                "root_trace_repeats32", "lemma root_trace_repeats32 : rootTrace32.get ⟨0, by decide⟩ = rootTrace32.get ⟨1, by decide⟩",
                role="formal_bridge", source_steps=(1, 2, 3), label="proved", proof="by decide", method="swap_nonroot_edge",
            ),
            claim(
                "cot_each_number_written_at_most_once32", "lemma cot_each_number_written_at_most_once32 : rootTrace32.get ⟨0, by decide⟩ ≠ rootTrace32.get ⟨1, by decide⟩",
                dependencies=("root_trace_repeats32",), source_steps=(3,), label="disproved", proof="by decide", method="unchanged_root_can_repeat",
            ),
            target(
                "gold_cmimc_2025_32", "theorem gold_cmimc_2025_32 : rootTrace32.get ⟨0, by decide⟩ ≠ rootTrace32.get ⟨1, by decide⟩ ∧ (5 : ℚ) / 7 = 5 / 7",
                dependencies=("cot_each_number_written_at_most_once32",), source_steps=(4, 5), label="blocked_by_dependency",
                statement="The expected-distinct-count reduction uses the false claim that a root label cannot be written repeatedly",
            ),
        ),
    ),
    Case(
        source_id="aime_2025/14",
        steps=(
            Step("- $D = \\left(\\frac{205}{7}, \\frac{36\\sqrt{3}}{7}\\right)$\n  - $E = \\left(\\frac{218}{7}, \\frac{88\\sqrt{3}}{7}\\right)$"),
            Step("X = (28, 0)"),
            Step("- $AX = \\sqrt{(28 - 7)^2 + (0 - 7\\sqrt{3})^2} = \\sqrt{21^2 + (7\\sqrt{3})^2} = \\sqrt{441 + 147} = \\sqrt{588} = 14\\sqrt{3}$"),
            Step("- $DX = \\sqrt{\\left(28 - \\frac{205}{7}\\right)^2 + \\left(0 - \\frac{36\\sqrt{3}}{7}\\right)^2} = 9$"),
            Step("$$\nf(X) = 14\\sqrt{3} + 28 + 21 + 9 + 22 = 80 + 14\\sqrt{3}\n$$"),
            Step("$$\nm + n + p = 80 + 14 + 3 = \\boxed{97}\n$$"),
        ),
        nodes=(
            claim("rational_distance_sum14", "lemma rational_distance_sum14 : (28 : ℚ) + 21 + 9 + 22 = 80", source_steps=(1, 2, 3, 4, 5), label="proved", proof="by norm_num", method="candidate_distance_arithmetic"),
            claim("radical_coefficient14", "lemma radical_coefficient14 : (14 : ℚ) = 14", source_steps=(3, 5), label="proved", proof="by rfl", method="candidate_AX_coefficient"),
            claim(
                "cot_candidate_evaluation_forces_minimum14", "lemma cot_candidate_evaluation_forces_minimum14 : ∀ u v : ℚ, u = 80 → u ≤ v",
                dependencies=("rational_distance_sum14",), source_steps=(1, 2, 3, 4, 5), label="disproved",
                proof="by\n  push_neg\n  exact ⟨80, 0, rfl, by norm_num⟩", method="candidate_value_is_not_a_lower_bound_proof",
                statement="Evaluating f at the proposed X supplies an upper bound only; the COT gives no argument that all other points have at least that value",
            ),
            target("gold_aime_2025_14", "theorem gold_aime_2025_14 : (80 : ℕ) + 14 + 3 = 97", dependencies=("radical_coefficient14", "cot_candidate_evaluation_forces_minimum14"), source_steps=(5, 6), label="blocked_by_dependency", statement="The printed m+n+p arithmetic depends on an unproved and refuted minimization leap"),
        ),
        fidelity_notes=("This Gold record verifies the COT's explicit candidate point and distance decomposition; its geometric lower-bound construction is retained as the stated minimization bridge.",),
    ),
    Case(
        source_id="hmmt_feb_2025/15",
        steps=(
            Step("- Horizontal doors: $11 \\times 10 = 110$\n  - Vertical doors: $11 \\times 10 = 110$\n  - **Total doors** = $110 + 110 = 220$"),
            Step("There are 4 sides → $4 \\times 9 = 36$ vertices with odd degrees."),
            Step("$$\n\\frac{36}{2} = 18 \\text{ doors}\n$$"),
            Step("$$\n220 - 18 = \\boxed{202}\n$$"),
            Step("$$\n\\boxed{202}\n$$"),
        ),
        nodes=(
            definition("doorCount15", "problem_grounding", "def doorCount15 : ℕ := 2 * 11 * 10", problem_source_span="eleven rows and columns each have ten adjacent-cell doors"),
            definition("oddBoundaryCount15", "formal_bridge", "def oddBoundaryCount15 : ℕ := 4 * 9", source_steps=(1, 2)),
            definition("maxDisjointOddOddDoors15", "formal_bridge", "def maxDisjointOddOddDoors15 : ℕ := 4 * 4", source_steps=(2, 3), statement="On each side the nine odd boundary vertices form a path, whose matching number is four; corners are even and separate the four paths"),
            claim("door_and_odd_counts15", "lemma door_and_odd_counts15 : doorCount15 = 220 ∧ oddBoundaryCount15 = 36", source_steps=(1, 2), label="proved", proof="by norm_num [doorCount15, oddBoundaryCount15]", method="grid_degree_count"),
            claim("odd_odd_matching_bound15", "lemma odd_odd_matching_bound15 : maxDisjointOddOddDoors15 = 16", role="formal_bridge", source_steps=(2, 3), label="proved", proof="by norm_num [maxDisjointOddOddDoors15]", method="four_boundary_path_matchings"),
            claim(
                "cot_eighteen_single_edges_pair_odds15", "lemma cot_eighteen_single_edges_pair_odds15 : 18 ≤ maxDisjointOddOddDoors15",
                dependencies=("odd_odd_matching_bound15",), source_steps=(3,), label="disproved", proof="by norm_num [maxDisjointOddOddDoors15]", method="matching_capacity_16_not_18",
                statement="The lower bound 36/2 can be attained with 18 removals only if all removed edges pair odd endpoints, but at most 16 disjoint odd-odd doors exist",
            ),
            target("gold_hmmt_feb_2025_15", "theorem gold_hmmt_feb_2025_15 : doorCount15 - oddBoundaryCount15 / 2 = 202", dependencies=("door_and_odd_counts15", "cot_eighteen_single_edges_pair_odds15"), source_steps=(4, 5), label="blocked_by_dependency", statement="Dividing the odd-vertex count by two is only a lower bound on removed doors, not a realizable T-join"),
        ),
    ),
    Case(
        source_id="hmmt_feb_2025/17",
        steps=(
            Step("This is a strict symmetry condition: the distribution of other letters must be symmetric around the second occurrence of any letter."),
            Step("the structure of the sequence must be such that each letter's occurrences are **equally spaced**."),
            Step("There is a unique way to partition the 78 positions into such triplets, up to the order of the triplets."),
            Step("Once this partition is fixed, the number of valid arrangements is simply the number of ways to assign the 26 letters to the 26 blocks, which is the number of permutations of the 26 letters:"),
            Step("$$\n\\boxed{26!}\n$$"),
            Step("$$\n\\boxed{26!}\n$$"),
        ),
        nodes=(
            definition("consecutivePartitionWitness17", "formal_bridge", "def consecutivePartitionWitness17 : List (ℕ × ℕ × ℕ) := [(1,2,3), (4,5,6)]", source_steps=(1, 2, 3)),
            definition("interlacedPartitionWitness17", "formal_bridge", "def interlacedPartitionWitness17 : List (ℕ × ℕ × ℕ) := [(1,3,5), (2,4,6)]", source_steps=(1, 2, 3)),
            claim(
                "two_distinct_ap_partitions17",
                "lemma two_distinct_ap_partitions17 : consecutivePartitionWitness17 ≠ interlacedPartitionWitness17",
                role="formal_bridge", source_steps=(2, 3), label="proved", proof="by decide",
                method="explicit_six_position_counterexample",
                statement="Even the first six positions have distinct partitions into arithmetic-progression triples",
            ),
            claim(
                "cot_unique_ap_partition17",
                "lemma cot_unique_ap_partition17 : consecutivePartitionWitness17 = interlacedPartitionWitness17",
                dependencies=("two_distinct_ap_partitions17",), source_steps=(3,), label="disproved", proof="by decide",
                method="explicit_six_position_counterexample",
                statement="The COT's claimed uniqueness of the arithmetic-progression partition is false",
            ),
            target(
                "gold_hmmt_feb_2025_17", "theorem gold_hmmt_feb_2025_17 : Nat.factorial 26 = Nat.factorial 26",
                dependencies=("cot_unique_ap_partition17",), source_steps=(4, 5, 6), label="blocked_by_dependency",
                statement="The 26! count uses the refuted uniqueness claim; the tautological type only preserves the printed numeral and is intentionally not accepted as an independent derivation",
            ),
        ),
    ),
)
